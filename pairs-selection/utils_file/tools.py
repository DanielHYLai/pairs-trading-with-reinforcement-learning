# Import the necessary packages
import pickle


def write_cache_file(obj: any, file_name: str):
    """
    Save the specified object as a pickle file.

    Parameters:
    ----------
    obj : any
        Objects to be saved as pickle files

    file_name : str
        The name of pickle file
    """

    with open(f"{file_name}", "wb") as file:
        pickle.dump(obj, file)



def load_cache_file(file_name: str):
    """
    Read the specified pickle file and export it.

    Parameters:
    ----------
    file_name : str
        The name of pickle file

    Returns:
    -------
    result : any
        Export objects stored in this file
    """

    with open(f"{file_name}", "rb") as file:
        result = pickle.load(file)

    return result


def show_elapsed_time(end_time: float):
    """
    Show the time elapsed in hours, minutes, and seconds.

    Parameters
    ----------
    end_time : float
        Input the time elapsed
    """

    seconds = end_time % 60
    minutes = int((end_time % 3600) // 60)
    hours   = int(end_time // 3600)

    print(f"Time elapsed: {hours} hours {minutes} minutes {seconds: .2f} seconds")
