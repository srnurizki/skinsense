# Import Libraries
from langchain_core.messages import HumanMessage
from agent.orchestrator import app

# Print Messages
def runMessages(messages, seen_count):
    for msg in messages[seen_count:]:
        role = type(msg).__name__
        content = getattr(msg, 'content', '')
        tool_calls = getattr(msg, 'tool_calls', None)
        print(f'[{role}]')
        if content:
            print(content)
        if tool_calls:
            print('tool_calls:', tool_calls)
        print()

# Run Session
def main():
    messages = []
    print('Agent is Ready. Ctrl+C to Exit.\n')

    while True:
        try:
            user_input = input('You: ')
        except KeyboardInterrupt:
            print('\nSession ended.')
            break

        if not user_input.strip():
            continue

        messages.append(HumanMessage(content=user_input))
        seen_count = len(messages)

        try:
            result = app.invoke({'messages': messages})
        except KeyboardInterrupt:
            print('\nSession ended.')
            break

        messages = result['messages']
        runMessages(messages, seen_count)

# Init
if __name__ == '__main__':
    main()