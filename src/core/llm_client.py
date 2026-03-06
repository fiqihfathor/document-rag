from openai import AsyncOpenAI
from src.config import settings
from src.utils.logger import logger

client = AsyncOpenAI(
    api_key="dummy",
    base_url=f"{settings.LLM_BASE_URL}/v1",
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided documents.

Rules:
- Answer ONLY based on the given context
- If the answer is not found in the context, respond with "The information was not found in the provided documents"
- Answer in the same language as the user's question
- Cite the source filename and page number when relevant
- Do not fabricate or infer information outside the given context"""


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[File: {c['filename']} | Page: {c['page_number']}]\n{' '.join(c['text'].split()[:200])}"
        for c in chunks
    )

async def generate(query: str, chunks: list[dict]) -> str:
    context = build_context(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    prompt_used = messages[1]["content"]
    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
    )

    answer = response.choices[0].message.content
    logger.debug(f"Generated answer — {len(answer)} chars")
    return{
        "answer": answer,
        "prompt_used": prompt_used,
        "token_usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }}
