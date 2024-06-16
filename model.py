import os

####################################################################################################

local_llm = 'llama3'

####################################################################################################

# Создаем векторную базу данных

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import FireCrawlLoader,CSVLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.docstore.document import Document

# Получение списка файлов и папок в указанной директории
files = os.listdir('docs')

local = []

for file in files:
    local.append(f"docs/{file}")

# loader = CSVLoader(file_path = '/Users/admin/Desktop/RosAtom_final.csv').load()

#docs = [FireCrawlLoader(api_key = 'fc-8e350ad629b84f9f87d9c0cc6879fa02',url = url,mode = 'scrape').load() for url in urls] # Файловый зыгрузчик, scrape означает, что мы будем создавать сценарий для каждого url адреса
#FireCrawlLoader()

docs = [UnstructuredPDFLoader(file_path = local_path).load() for local_path in local if local_path]

docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250,chunk_overlap = 0
)

doc_splits = text_splitter.split_documents(docs_list)

# Очистка данных от нечитабельных токенов
filtered_docs = []
for doc in doc_splits:
  if isinstance(doc,Document) and hasattr(doc,'metadata'):
    clean_metadata = {k: v for k,v in doc.metadata.items() if isinstance(v,(str,int,float,bool))}
    filtered_docs.append(Document(page_content = doc.page_content,metadata = clean_metadata))

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}

vectorstore = Chroma.from_documents( #Создание базы данных
    documents = filtered_docs,
    collection_name = 'rag-chroma',
    embedding = GPT4AllEmbeddings(model_name = model_name,
                                  gpt4all_kwargs = gpt4all_kwargs))


retviever = vectorstore.as_retriever() # создаем поисковик для векторного поиска базы данных

####################################################################################################

# Теперь мы хотим понять имеет ли поисковик какое-то дело к вопросу
# поэтому мы создаем способ оченки поисковика

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOllama(model = local_llm,format = 'json',temperature = 0)

prompt = PromptTemplate(
      template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stingent test.
      The goal is to filter out erroneous retrievals. In Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
      In Provide the binary score as a JSON with a single key 'score' and no premable or explaination. 
      <leot_id/><|start_header_id|>user<|end_header_id|>
      Here is the retrieved document: \n\n {documents} \n\n
      Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
      """,
      input_variables = ['question','documents']
)

retrieval_grader = prompt | llm | JsonOutputParser()
question = 'Полученные переработчиком давальческие материалы можно перемещать между складами?' # данный вопрос является актуальным, поэтому она выдаст ответ да
docs = retviever.invoke(question) # Реализуем поиск по базе данных
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({'question':question,'documents':doc_txt}))

####################################################################################################

# Далее если вопрос явлется актуальным, то мы уже генерируем ответ

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>systems<|end_header_id|> Вы являетесь помощником при выполнении заданий по поиску ответов на вопросы.
    Используйте приведенные ниже фрагменты из извлеченного контекста, чтобы ответить на вопрос. Если вы не знаете ответа, просто скажите, что вы не знаете.
    Используйте максимум столько предложений, сколько понадобится, но чтоб было не больше пяти предложений.
    Сгенерируй полный ответ, собрав информацию со всех источников на вопрос из самого файла на Русском языке.
    Также сгенерируй название файла, в котором ты брал информацию.
    Если к тебе вернулись снова, то перефразируй ответ <|eot_id|><|start_header_id|>user<|end_header_id|>
    Вопрос: {question}
    Контекст: {context}
    Вот ответ на ваш запрос: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question","documents"])

llm = ChatOllama(model = local_llm,temperature = 0)

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()
question = 'Полученные переработчиком давальческие материалы можно перемещать между складами? Подробнее опиши сам процесс'

generation = rag_chain.invoke({'context': docs ,'question': question})
print(generation)

####################################################################################################

llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt
prompt = PromptTemplate(
  template = """ <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an answer is grounded in / supported by a set of facts.
  Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts.
  Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>uses<|end_header_id|>
  Here are the facts:
  \n -------- \n
  {documents}
  \n -------- \n
  Here is the answer: {generation} <|leot_id|><|start_header_id >assistant<lend_header_id|>""",
  input_variables=("generation", "documents"),
)
hallucination_grader = prompt | llm | JsonOutputParser ()
hallucination_grader.invoke({"documents": docs,"generation": generation})

####################################################################################################

llm = ChatOllama(model = local_llm,format = 'json',temperature = 0)

prompt = PromptTemplate(
    template = """<|begin_of_text|><|start_header_id|>systems|end_header_id|> You are a grader assessing whether an answer is useful to resolve a question.
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    <eot_id|><|start_header_id|>user<|end_header_id|> 
    Here is the answer:
    \n-------- \n
    {generation}
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()
answer_grader.invoke({'question': question,'generation':generation})

####################################################################################################

os.environ['TAVILY_API_KEY'] = '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'

from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k = 3)

####################################################################################################

from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
  """
  Тут мы создаем значения земли, то есть это те значения,
  С которыми мы хотим поделиться на различных этапах
  В нашем случае у нас будут:

  question: вопрос
  generation: вывод LLM
  web_search: вывод веб-поиска
  documents: Список документов
  """
  question: str
  generation: str
  web_search: str
  documents: List[str]

from langchain.schema import Document

def retvieve(state):
    """
    Данный узел просто нужен для обращения к базе данных
    Мы выводим документ, который содержит контекст нашего вопроса
    """
    print('Обращаемся к базе данных')
    question = state['question']
    documents = retviever.invoke(question)
    return {'documents': documents, 'question': question} # переписываем глобальное состояние

def grade_documents(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
      state (dict): The current graph state

    Returns:
      state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("--Проверка на релевантность документа--")
    question = state['question']
    documents = state['documents']

    filtered_docs = []
    web_search = 'No'
    for d in documents:
      score = retrieval_grader.invoke({'question':question,'documents':d.page_content})
      grade = score['score']

      if grade.lower() == 'yes':
        filtered_docs.append(d)
      else:
        print('--Документ не является релевантным')
        web_search = 'Yes'
        continue
      return {'documents':filtered_docs,'question':question,'web_search':web_search}

def generate(state):
    """
    Проверяем нужно ли выполнять web поиск или сразу генерировать ответ
    """
    print('--Генерируем ответ--')
    question = state['question']
    documents = state['documents']

    generation = rag_chain.invoke({'context':documents,'question':question})
    return {'documents':documents, 'question':question, 'generation':generation}

def web_search(state):
    print('--WEB SEARCH--')
    question = state['question']
    documents = state['documents']

    docs = web_search_tool.invoke({'query':question})
    web_results = '\n'.join([d['content'] for d in docs])
    web_results = Document(page_content = web_results)
    if documents is not None:
      documents.append(web_results)
    else:
      documents = [web_results]
    return {'documents':documents, 'question': question}

def deside_to_generate(state):
      print('--Очениваем утвержденные документы--')
      question = state['question']
      web_search = state['web_search']
      filtered_documents = state['documents']

      if web_search == 'Yes':
          print('--Все документы не для вопроса явялются нерелевантными')
          return 'websearch'
      else:
          print('--Генерируем--')
          return 'generate'

def check_halluciation(state):
      print('Проверка на галлюцинации')
      question = state['question']
      documents = state['documents']
      generation = state['generation']

      score = hallucination_grader.invoke({'documents': documents,'generation':generation})
      grade = score['score']

      if grade == 'yes':
        score = answer_grader.invoke({'question': question,'generation': generation})
        grade = score['score']
        if grade == 'yes':
          return 'useful'
        else:
          return 'not useful'
      else:
        return 'not supported'

####################################################################################################

from langgraph.graph import StateGraph
from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

workflow.add_node('websearch',web_search)
workflow.add_node('retvieve',retvieve)
workflow.add_node('grade_documents',grade_documents)
workflow.add_node('generate',generate)

workflow.set_entry_point('retvieve')
workflow.add_edge('retvieve','grade_documents')
workflow.add_conditional_edges(
    'grade_documents',
    deside_to_generate,
    {
        'websearch': 'websearch',
        'generate': 'generate'
    },
)
workflow.add_edge('websearch','generate')
workflow.add_conditional_edges(
    'generate',
    check_halluciation,
    {
        'not_supported':'generate',
        'useful': END,
        'not useful': 'websearch'
    },
)

####################################################################################################
from pprint import pprint

app = workflow.compile()

def startDialog(question: str):
  if (question == ""): return None
  
  inputs = {'question': question}
  for out in app.stream(inputs):
      for key,value in out.items():
          pprint(f'Finished running: {key}:')

  return value['generation']
