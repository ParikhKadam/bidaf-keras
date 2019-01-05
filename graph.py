from keras.layers import Input, TimeDistributed, LSTM, Bidirectional, Lambda
from keras.models import Model
from layers import Highway, Similarity, C2QAttention, Q2CAttention, MergedContext, SpanBegin, SpanEnd, CombineOutputs
from scripts import negative_avg_log_error, BatchGenerator
import pickle


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

model.compile(loss=negative_avg_log_error, optimizer='adadelta', metrics=[])

train_generator = BatchGenerator('train')
validate_generator = BatchGenerator('dev')

epochs = 40

history = model.fit_generator(train_generator, epochs=1, validation_data=validate_generator, shuffle=False)
with open('history', 'wb') as f:
    pickle.dump(history.history,f)
model.save('bidaf_0')

for i in range(1, epochs):
    with open('history', 'rb') as f:
        old_history = pickle.load(f)

    history = model.fit_generator(train_generator, epochs=1, validation_data=validate_generator, shuffle=False)
    old_history['loss'].extend(history.history['loss'])
    old_history['val_loss'].extend(history.history['val_loss'])
    
    with open('history', 'wb') as f:
        pickle.dump(old_history, f)    
    
    model.save('bidaf_{}'.format(i))