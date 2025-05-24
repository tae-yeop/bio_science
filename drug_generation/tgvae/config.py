from pydantic import BaseModel

class Config(BaseModel):
    wandb_project_name:str
    wandb_key:str
    wandb_host:str
    # Model hyperparameters
    dim_encoder:int, default=512
    dim_decoder:int, default=512)
    dim_latent:int, default=256)
    dim_encoder_ff:int, default=512)
    dim_decoder_ff', type=int, default=512)
    num_encoder_layer', type=int, default=4)
    parser.add_argument('-ndl', '--num_decoder_layer', type=int, default=4)
    parser.add_argument('-neh', '--num_encoder_head', type=int, default=1)
    parser.add_argument('-ndh', '--num_decoder_head', type=int, default=16)
    parser.add_argument('-doe', '--dropout_encoder', type=float, default=0.3)
    parser.add_argument('-dog', '--dropout_gat', type=float, default=0.3)
    parser.add_argument('-dod', '--dropout_decoder', type=float, default=0.3)

    # Training hyperparameters
    batch_size:int, default=128)
    epoch:int, default=40)
    gradient_clipping:float, default=5.0)

