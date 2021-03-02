from sklearn.model_selection import train_test_split

from used_car_price.config import config
from used_car_price.processing.data_management import load_dataset, save_dataset

if __name__ == '__main__':
	# read the original data
	df = load_dataset(file_name=config.RAW_DATA_FILE)

	# Split original data into train and test data using 80-20 rule
	train, test = train_test_split(df, test_size=0.20, random_state=config.RANDOM_STATE, shuffle=True)

	# save train and test
	save_dataset(df=train, file_name=config.TRAIN_DATA_FILE, index=0, compression='gzip')
	save_dataset(df=test, file_name=config.TEST_DATA_FILE, index=0, compression='gzip')
