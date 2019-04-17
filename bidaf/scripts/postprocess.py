def get_best_span(span_begin_probs, span_end_probs, context, squad_version, max_span_length):
    if len(span_begin_probs.shape) > 2 or len(span_end_probs.shape) > 2:
        raise ValueError("Input shapes must be (X,) or (1,X)")
    if len(span_begin_probs.shape) == 2:
        assert span_begin_probs.shape[0] == 1, "2D input must have an initial dimension of 1"
        span_begin_probs = span_begin_probs.flatten()
    if len(span_end_probs.shape) == 2:
        assert span_end_probs.shape[0] == 1, "2D input must have an initial dimension of 1"
        span_end_probs = span_end_probs.flatten()

    max_span_probability = 0
    best_word_span = (0, 1)

    for i, val1 in enumerate(span_begin_probs):
        if squad_version == 2.0 and i == 0:
            continue

        for j, val2 in enumerate(span_end_probs):
            if j > len(context) - 1:
                break

            if (squad_version == 2.0 and j == 0) or (j < i):
                continue

            if (j - i) >= max_span_length:
                break

            if val1 * val2 > max_span_probability:
                best_word_span = (i, j)
                max_span_probability = val1 * val2

    if squad_version == 2.0:
        if span_begin_probs[0] * span_end_probs[0] > max_span_probability:
            best_word_span = (0, 0)
            max_span_probability = span_begin_probs[0] * span_end_probs[0]

    return best_word_span, max_span_probability
