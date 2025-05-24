from models.ribonanzanet.network import RibonanzaNet

class finetuned_RibonanazNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        config.dropout = 0.2
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load(
                config.pretrained_path, 
                map_location='cpu',
                weights_only=True
            ))
        self.dropout = nn.Dropout(0.0)
        self.xyz_predictor = nn.Linear(256, 3)

    def forward(self, src):
        sequence_features, pairwise_features = self.get_embeddings(src, )