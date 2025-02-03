import asyncio
import io
import re
import httpx
from openai import AsyncOpenAI
from langchain.prompts import PromptTemplate
from pymongo import MongoClient

class LLM:
    def __init__(self, config, prompts):
        self.config = config
        self.client = self._init_openai_client()
        self.mongo_client = self._init_mongo_client()
        self.system_prompt = self._load_prompt(prompts['system_prompt'])
        self.user_prompt_template = self._create_template(prompts['user_prompt'])
        self.semaphore = asyncio.Semaphore(int(config.llm.get('max_concurrent_requests', 3)))

    def _init_openai_client(self):
        return AsyncOpenAI(
            #base_url=self.config.llm['base_url'],
            api_key=self.config.api_key,
            timeout=30,
            max_retries=2
        )

    def _init_mongo_client(self):
        return MongoClient(self.config.Mongo['connexion_string'])

    def _load_prompt(self, filename):
        with io.open(f"{self.config.prompts['dir']}/{filename}", 'r', encoding='utf-8') as f:
            return f.read()

    def _create_template(self, template_file):
        content = self._load_prompt(template_file)
        # Vérification explicite des variables
        required_vars = {'chunk', 'previous_analysis'}
        found_vars = set(re.findall(r'\{(.*?)\}', content))
        
        if found_vars != required_vars:
            raise ValueError(f"Variables manquantes dans le template. Requises: {required_vars}, Trouvées: {found_vars}")
        
        return PromptTemplate(
            input_variables=list(required_vars),
            template=content
        )

    async def _process_chunk(self, chunk, previous_analysis=None):
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.llm['name'],
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.user_prompt_template.format(
                            chunk=chunk,
                            previous_analysis=previous_analysis or "Première analyse"
                        )}
                    ],
                    temperature=0.2,
                    max_tokens=int(self.config.llm['max_tokens'])
                )
                r = response.choices[0].message.content.strip()
                print(r)
                return r
            except httpx.HTTPError:
                return None

    async def refine_analysis(self, chunks):
        analysis = None
        for chunk_batch in chunks:
            # Concaténation sécurisée des textes
            batch_text = "\n---\n".join(
                doc.get('input_text', '') 
                for doc in chunk_batch 
                if isinstance(doc, dict)
            )
            
            analysis = await self._process_chunk(
                chunk=batch_text,
                previous_analysis=analysis
            )
        return analysis