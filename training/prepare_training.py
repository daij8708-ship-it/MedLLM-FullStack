import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workspace import HOME as ROOT
config={
 'model_name_or_path':str(ROOT/'models/DeepSeek-R1-Distill-Qwen-1.5B'),
 'cache_dir':str(ROOT/'cache/lf'),'trust_remote_code':False,'stage':'sft','do_train':True,
 'finetuning_type':'lora','lora_rank':8,'lora_alpha':16,'lora_dropout':0.05,'lora_target':'all',
 'quantization_bit':4,'quantization_method':'bitsandbytes','template':'deepseek_evidence',
 'dataset_dir':str(ROOT/'data/evidence-lf-v1'),'dataset':'evidence_train','eval_dataset':'evidence_validation',
 'cutoff_len':512,'preprocessing_num_workers':1,'dataloader_num_workers':0,'dataloader_pin_memory':False,
 'output_dir':str(ROOT/'runs/evidence-lf-v1'),'logging_steps':10,'save_strategy':'no',
 'report_to':'none','overwrite_output_dir':False,'per_device_train_batch_size':1,
 'per_device_eval_batch_size':1,'gradient_accumulation_steps':4,'learning_rate':7e-5,
 'max_steps':120,'warmup_steps':8,'lr_scheduler_type':'cosine','bf16':True,
 'gradient_checkpointing':True,'optim':'adamw_torch','seed':42,'data_seed':42,
 'eval_strategy':'no','flash_attn':'disabled','plot_loss':False,
}
(ROOT/'evidence-lf-v1.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print(ROOT/'evidence-lf-v1.json')
