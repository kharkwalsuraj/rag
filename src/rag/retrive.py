import ollama

from rag.lib import Embed 


def retrive_answers(query: str, ranker_model_id: str, lmm_model_id:str):
    

    embed = Embed()
    embeded_query = embed.embed_query(query)
