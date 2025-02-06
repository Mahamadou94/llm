import asyncio
import os
import io
import re
import json
import httpx
from openai import AsyncOpenAI
from langchain.prompts import PromptTemplate

class LLM:
    def __init__(self, config, prompts, chunks):
        self.config = config
        self.client = self.set_client()
        self.system_prompt = self.get_prompt_from_file(prompts['system_prompt'])
        self.user_prompt_template = self.create_template(prompts['user_prompt'])
        self.semaphore = asyncio.Semaphore(int(config.llm.get('max_concurrent_requests', 3)))
        self.chunks = chunks

    def set_client(self):
        return AsyncOpenAI(
            api_key=self.config.api_key,
            timeout=60,
            max_retries=3
        )

    def get_prompt_from_file(self, file_name):
        with io.open(os.path.join(self.config.prompts['dir'], file_name), 'r', encoding='utf-8') as f:
            return f.read()

    def create_template(self, template_file):
        content = self.get_prompt_from_file(template_file)
        input_variables = re.findall(r'\{(.*?)\}', content)
        return PromptTemplate(input_variables=input_variables, template=content)

    async def send_request_with_retry(self, user_prompt):
        for attempt in range(7):  # Tentatives jusqu'à 7 fois
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.llm['name'],
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=False,
                    max_tokens=int(self.config.llm['max_tokens'])
                )
                #if response.status_code == 200:
                r = response.choices[0].message.content.strip()
                print(f'Ceci est la réponse llm !!!!!!!!!! {"/n", r}')
                return r
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait_time = float(e.response.headers['Retry-After'])
                    print(f"Rate limit exceeded, retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"HTTP error occurred: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

    async def infer(self, chunk):
        async with self.semaphore:
            user_prompt = self.user_prompt_template.format(chunk=chunk)
            return await self.send_request_with_retry(user_prompt)

    async def run(self):
        tasks = [self.infer(chunk) for chunk in self.chunks]
        return await asyncio.gather(*tasks)
