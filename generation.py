import asyncio
from llm import LLM
from config_parser import Parser
from pymongo import ASCENDING
from datetime import datetime

class ReviewProcessor:
    def __init__(self, config_path):
        self.config = Parser(config_path)
        self.llm = LLM(self.config, {
            'system_prompt': self.config.prompts['system_prompt'],
            'user_prompt': self.config.prompts['user_prompt']
        })
        
    def _chunk_documents(self, documents, batch_size=5):
        """Grouper les documents par lots avec suivi de contexte"""
        for i in range(0, len(documents), batch_size):
            yield documents[i:i + batch_size]

    def _save_to_file(self, analysis, chunk_id):
        """Save the analysis to a text file with a timestamp and chunk ID"""
        filename = f"analysis_chunk_{chunk_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        with open(filename, 'w') as file:
            file.write(analysis)
        print(f"Analysis for chunk {chunk_id} saved to {filename}")

    async def process(self):
        db = self.llm.mongo_client[self.config.Mongo['data_base']]
        collection = db[self.config.Mongo['collection']]
        
        # Récupération synchrone des données
        documents = list(collection.find().sort('timestamp', ASCENDING))
        
        # Découpage intelligent avec contexte
        chunks = self._chunk_documents(documents, batch_size=int(self.config.processing['batch_size']))
        
        # Traitement pipeline and save each chunk's analysis immediately
        chunk_id = 0
        for chunk in chunks:
            final_analysis = await self.llm.refine_analysis(chunk)
            self._save_to_file(final_analysis, chunk_id)
            chunk_id += 1

        # Optionally, you could also save a final report if needed
        report_collection = db['analysis_reports']
        report_collection.insert_one({
            'timestamp': datetime.now(),
            'analysis': 'Final Analysis Completed',
            'processed_count': len(documents)
        })
        
        return "Processing complete."

if __name__ == "__main__":
    processor = ReviewProcessor("config.ini")
    asyncio.run(processor.process())
