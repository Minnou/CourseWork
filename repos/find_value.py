import pandas as pd 

class DataFrameIterator:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.row_iterator = dataframe.iterrows()
    
    def __iter__(self):
        return self

    def __next__(self):
        """Get next row in given dataframe

        Returns:
            next row in given dataframe
        """
        return next(self.row_iterator)

def create_dataset_from_files(files: list):
    """Create pandas dataframe from list of files

    Args:
        files: list of files
    Returns:
        pandas dataframe object
    """
    if len(files) == 1:
        result_data_frame = pd.read_csv(files[0], sep=";")
        return result_data_frame
    elif len(files ) >= 2:
        data_frames = []
        for filename in files:
            data_frames.append(pd.read_csv(filename, sep=";"))
        if len(files) == 2:
            if (data_frames[0].shape[1] == 1 or data_frames[1].shape[1] == 1):
                if data_frames[0].shape[1] == 1:
                    result_data_frame = pd.concat([data_frames[0], data_frames[1]], axis=1)
                else:
                    result_data_frame = pd.concat([data_frames[1], data_frames[0]], axis=1)
                return result_data_frame
        result_data_frame = pd.DataFrame()
        for data_frame in data_frames:
            result_data_frame = pd.concat([result_data_frame, data_frame], axis=0)
        return result_data_frame

def get_value(date: str, files: list):
    """Get value for the given date from the given file(s)

    Args:
        date: date in yyyy-mm-dd format
        files: list of files
    Returns:
        Value if found None if not
    """
    data_frame = create_dataset_from_files(files)
    for index, line in DataFrameIterator(data_frame):
        if line.iloc[0] == date:
            return line.iloc[1]
    return None

def main():
    print("Введите пути до файлов через пробел (./dataset.csv): ", end="")
    files = input()
    if files.strip() == "":
        files = "./dataset.csv"
        #files = "./x.csv ./y.csv"
    files = files.split(" ")
    
    print("Введите дату (2007-04-20): ", end="")
    date = input()
    if date.strip() == "":
        date = "2007-04-20"
    
    print(get_value(date=date, files=files))
    create_dataset_from_files

if __name__ == '__main__':
    main()