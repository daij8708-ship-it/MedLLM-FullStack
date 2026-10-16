"""Register the existing DeepSeek answer-only framing, then use real LF SFT."""
import json,sys
from llamafactory.data.template import register_template
from llamafactory.data.formatter import StringFormatter,EmptyFormatter
from llamafactory.train.tuner import run_exp
register_template(name='deepseek_evidence',
 format_user=StringFormatter(slots=['<｜User｜>{{content}}<｜Assistant｜><think>\n</think>\n\n']),
 format_prefix=EmptyFormatter(slots=[{'bos_token'}]),replace_jinja_template=True)
if __name__=='__main__':
 run_exp(json.load(open(sys.argv[1],encoding='utf-8')))
