CHAT_TEMPLATE = """{{ bos_token }}{%- if tools %}{{ raise_exception('DeepSeek-VL2 tool calling is unsupported') }}{% endif -%}
{%- for message in messages -%}
{%- set content = namespace(text='') -%}
{%- if message['content'] is string -%}
{%- set content.text = message['content'] -%}
{%- else -%}
{%- for part in message['content'] -%}
{%- if part['type'] == 'text' -%}{%- set content.text = content.text + part['text'] -%}
{%- elif part['type'] in ['image', 'image_url'] -%}{%- set content.text = content.text + '<image>\n' -%}
{%- else -%}{{ raise_exception('unsupported DeepSeek-VL2 content type') }}{%- endif -%}
{%- endfor -%}{%- endif -%}
{%- if message['role'] == 'system' -%}{{ content.text.strip() + '\n\n' }}
{%- elif message['role'] == 'user' -%}{{ '<|User|>: ' + content.text.strip() + '\n\n' }}
{%- elif message['role'] == 'assistant' -%}{{ '<|Assistant|>: ' + content.text.strip() + eos_token }}
{%- else -%}{{ raise_exception('unsupported DeepSeek-VL2 message role') }}{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{ '<|Assistant|>:' }}{%- endif -%}"""
