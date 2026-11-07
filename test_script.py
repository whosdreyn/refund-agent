import uuid
from agent import build_agent, ask
from trajectory_engine import from_langchain_messages

r = ask(build_agent(), "awa ici. Je veux rembourser un seul des deux câbles USB-C, je garde l autre.", str(uuid.uuid4()))
for etape in from_langchain_messages(r['messages']):
    print(f"Etapes : {etape.tool} , Arguments: {etape.arguments} , Résultat: {etape.result}")
