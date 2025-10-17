from langchain import hub
prompt = hub.pull("hwchase17/react")
prompt.pretty_print()