from speech_embedder_net import SpeechEmbedder
from Speaker_Verification.utils import mfccs_and_spec
import torch
from hparam import hparam as hp

class SpekerVerifier:
    """Class for loading the model and computing speech embeddings"""
    def __init__(
            self,
            model_path: str = './speech_id_checkpoint/ckpt_epoch_360_batch_id_28.pth'
            ) -> None:
        self.model = SpeechEmbedder()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def compute_embedding(self, audio_input):
        _, mel_db, _ = mfccs_and_spec(audio_input, wav_process = True)
        mel_db = torch.Tensor(mel_db)
        enrollment_batch = torch.reshape(mel_db, (1, mel_db.size(0), mel_db.size(1)))
        embedding = self.model(torch.Tensor(enrollment_batch))
        return embedding