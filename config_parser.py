import os 
from dotenv import load_dotenv  
from configparser import ConfigParser  


class Parser :
    def __init__(self, path_to_ini) :
        load_dotenv()
        parser = ConfigParser()
        self.api_key = os.getenv("API_KEY")
        parser.read_file(open(path_to_ini))
        for key, value in parser.items():
            setattr(self,key, dict(value.items()))