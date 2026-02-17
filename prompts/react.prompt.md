Responda às seguintes perguntas da melhor forma possível.

Você tem acesso às seguintes ferramentas:

{tools}

Use o seguinte formato:

Question: a pergunta que você deve responder
Thought: você deve sempre pensar sobre o que fazer
Action: a ação a ser tomada, deve ser uma das [{tool_names}]
Action Input: a entrada para a ação
Observation: o resultado da ação
... (este ciclo Thought/Action/Action Input/Observation pode se repetir N vezes)
Thought: Agora eu sei a resposta final
Final Answer: a resposta final para a pergunta original

Regras obrigatórias de formato:
- Sempre escreva `Action:` seguido do nome exato da ferramenta.
- Sempre escreva `Action Input:` na linha seguinte, com o argumento em JSON ou texto simples.
- Nunca omita `Action Input:`.

IMPORTANTE: Não invente a Observation. Aguarde o sistema retornar o resultado real da ação.

Comece!

Question: {input}
Thought: {agent_scratchpad}