from langfuse import observe
from litellm import embedding
from rich import print


@observe()
def main(text: str):
    response = embedding(
        model="jina_ai/jina-embeddings-v3",
        input=[text],
        metadata={
            "project": "hinbox",
            "session_id": "test_session_id",
            "trace_id": "test_trace_id",
        },
    )
    print(response)
    return response


# @observe()
# def hf_main(text: str):
#     response = embedding(
#         model="huggingface/jinaai/jina-embeddings-v3",
#         input=[text],
#         metadata={"project": "hinbox"},
#     )
#     print(response)
#     return response


if __name__ == "__main__":
    main("good morning from litellm")
    # hf_main("good morning from litellm")

"""
EmbeddingResponse(
    model='jina-embeddings-v3',
    data=[
        {
            'object': 'embedding',
            'index': 0,
            'embedding': [
                0.039701905,
                -0.043022938,
                0.044342972,
                -0.02652016,
                0.045551326,
                ETC ETC :)
                -0.0048891553,
                -0.023193313,
                -0.007748474,
                -0.04729483,
                0.011873731,
                -0.013572833,
                -0.023016868
            ]
        }
    ],
    object='list',
    usage=Usage(completion_tokens=0, prompt_tokens=8, total_tokens=8, completion_tokens_details=None, prompt_tokens_details=None)
)
"""
