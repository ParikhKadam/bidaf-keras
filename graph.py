from keras.layers import Input, TimeDistributed, LSTM, Bidirectional, Lambda
from keras.models import Model
from keras.optimizers import Adadelta
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import multi_gpu_model
from layers import Highway, Similarity, C2QAttention, Q2CAttention, MergedContext, SpanBegin, SpanEnd, CombineOutputs
from scripts import negative_avg_log_error, BatchGenerator
import pickle
import os


emdim = 600

question_input = Input(shape=(None, emdim), dtype='float32', name="question_input")
passage_input = Input(shape=(None, emdim), dtype='float32', name="passage_input")

num_highway_layers = 2

for i in range(num_highway_layers):
    highway_layer = Highway(name='highway_{}'.format(i))
    question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
    question_embedding = question_layer(question_input)
    passage_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_ptd")
    passage_embedding = passage_layer(passage_input)

encoder_layer = Bidirectional(LSTM(emdim, return_sequences=True), name='bidirectional_encoder')
encoded_question = encoder_layer(question_embedding)
encoded_passage = encoder_layer(passage_embedding)

similarity_matrix = Similarity(name='similarity_layer')([encoded_passage, encoded_question])

context_to_query_attention = C2QAttention(name='context_to_query_attention')([similarity_matrix, encoded_question])
query_to_context_attention = Q2CAttention(name='query_to_context_attention')([similarity_matrix, encoded_passage])

merged_context = MergedContext(name='merged_context')([encoded_passage, context_to_query_attention, query_to_context_attention])

num_decoders = 1

modeled_passage = merged_context
for i in range(num_decoders):
    hidden_layer = Bidirectional(LSTM(emdim, return_sequences=True), name='bidirectional_decoder_{}'.format(i))
    modeled_passage = hidden_layer(modeled_passage)

span_begin_probabilities = SpanBegin(name='span_begin')([merged_context, modeled_passage])
span_end_probabilities = SpanEnd(name='span_end')([encoded_passage, merged_context, modeled_passage, span_begin_probabilities])

output = CombineOutputs(name='combine_outputs')([span_begin_probabilities, span_end_probabilities])

model = Model([passage_input, question_input], [output])

model.summary()

try:
    model = multi_gpu_model(model)
except:
    pass

adadelta = Adadelta(lr=0.5)
model.compile(loss=negative_avg_log_error, optimizer='adadelta', metrics=[])

train_generator = BatchGenerator('train')
validate_generator = BatchGenerator('dev')

epochs = 40

saved_items_dir = os.path.join(os.path.dirname(__file__), 'saved_items')
if not os.path.exists(saved_items_dir):
    os.makedirs(saved_items_dir)

history_file = os.path.join(saved_items_dir, 'history.csv')
csv_logger = CSVLogger(history_file, append=True)

save_model_file = os.path.join(saved_items_dir, 'bidaf_{epoch:02d}.h5')
checkpointer = ModelCheckpoint(filepath=save_model_file, verbose=1)

history = model.fit_generator(train_generator, epochs=epochs, callbacks=[csv_logger, checkpointer], validation_data=validate_generator, shuffle=False)
