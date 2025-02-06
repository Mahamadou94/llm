import asyncio
from config_parser import Parser
from pymongo import MongoClient, ASCENDING
from api_call import LLM
import logging

# Configuration de la journalisation
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ReviewProcessor:
    def __init__(self, config_path):
        self.config = Parser(config_path)
        self.mongo_client = MongoClient(self.config.Mongo['connexion_string'])
        self.db = self.mongo_client[self.config.Mongo['data_base']]
        self.collection = self.db[self.config.Mongo['collection']]
        #logging.info("ReviewProcessor initialized and database connected.")

    def fetch_documents(self):
        #logging.info("Fetching documents from MongoDB.")
        return list(self.collection.find().sort('timestamp', ASCENDING))

    async def process_documents(self):
        documents = self.fetch_documents()
        texts = [doc['input_text'] for doc in documents]
        chunks = [texts[i:i + 500] for i in range(0, len(texts), 500)]
        llm = LLM(self.config, {'system_prompt': 'system_prompt.txt', 'user_prompt': 'user_prompt.txt'}, chunks)

        # Ouvrir les fichiers une seule fois pour tout le processus
        with open("complete_analysis2.txt", 'a') as all_results_file, open("llm_responses2.txt", 'a') as responses_file:
            for chunk in chunks:
                result = await llm.infer(chunk)
                # Écrire le chunk et son résultat dans le fichier 'complete_analysis.txt'
                chunk_text = " ".join(chunk)
                all_results_file.write(f"Chunk:\n{chunk_text}\n\nResult:\n\n{result}\n\n")
                all_results_file.write("=" * 40 + "\n")
                
                # Écrire uniquement la réponse LLM dans le fichier 'llm_responses.txt'
                responses_file.write(f"{result}\n")
                responses_file.write("=" * 40 + "\n")
                #logging.info(f"Results for a chunk have been saved.")

if __name__ == "__main__":
    processor = ReviewProcessor("config.ini")
    asyncio.run(processor.process_documents())
