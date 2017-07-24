
#TO TEST!!!!
training_decoder_input_talk = ["1","2","3","4","5","6"]
training_decoder_output_talk = ["1","2","3","4","5","6"]

max_sentence_length = 5

if len(training_decoder_input_talk) > max_sentence_length:
	training_decoder_input_talk = training_decoder_input_talk[:max_sentence_length+1]
	training_decoder_output_talk = training_decoder_output_talk[:max_sentence_length+1]
training_decoder_input_talk = training_decoder_input_talk[:len(training_decoder_input_talk)-1]
training_decoder_output_talk = training_decoder_output_talk[1:]

print training_decoder_input_talk
print training_decoder_output_talk