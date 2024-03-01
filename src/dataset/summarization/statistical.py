import pandas as pd

train_data = pd.read_csv('data/downstream/summazization/train.csv')
test_data = pd.read_csv('data/downstream/summazization/test.csv')

# print the # of rows for each data
print(f"Train data: {len(train_data)} rows")
print(f"Test data: {len(test_data)} rows")

# count the number of words for each file, for input_text and summary_text columns

train_data['input_text_num_words'] = train_data['input_text'].apply(lambda x: len(x.split()))
train_data['summary_text_num_words'] = train_data['summary_text'].apply(lambda x: len(x.split()))

test_data['input_text_num_words'] = test_data['input_text'].apply(lambda x: len(x.split()))
test_data['summary_text_num_words'] = test_data['summary_text'].apply(lambda x: len(x.split()))

# calculate the mean, min, max for input_text_num_words and summary_text_num_words columns
mean_input_text_num_words_train = train_data['input_text_num_words'].mean()
mean_summary_text_num_words_train = train_data['summary_text_num_words'].mean()

min_input_text_num_words_train = train_data['input_text_num_words'].min()
min_summary_text_num_words_train = train_data['summary_text_num_words'].min()

max_input_text_num_words_train = train_data['input_text_num_words'].max()
max_summary_text_num_words_train = train_data['summary_text_num_words'].max()

print(f"Train data: mean_input_text_num_words: {mean_input_text_num_words_train}, mean_summary_text_num_words: {mean_summary_text_num_words_train}, min_input_text_num_words: {min_input_text_num_words_train}, min_summary_text_num_words: {min_summary_text_num_words_train}, max_input_text_num_words: {max_input_text_num_words_train}, max_summary_text_num_words: {max_summary_text_num_words_train}")

mean_input_text_num_words_test = test_data['input_text_num_words'].mean()
mean_summary_text_num_words_test = test_data['summary_text_num_words'].mean()

min_input_text_num_words_test = test_data['input_text_num_words'].min()
min_summary_text_num_words_test = test_data['summary_text_num_words'].min()

max_input_text_num_words_test = test_data['input_text_num_words'].max()
max_summary_text_num_words_test = test_data['summary_text_num_words'].max()

print(f"Test data: mean_input_text_num_words: {mean_input_text_num_words_test}, mean_summary_text_num_words: {mean_summary_text_num_words_test}, min_input_text_num_words: {min_input_text_num_words_test}, min_summary_text_num_words: {min_summary_text_num_words_test}, max_input_text_num_words: {max_input_text_num_words_test}, max_summary_text_num_words: {max_summary_text_num_words_test}")