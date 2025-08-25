import importlib
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TTS_my"))

# 2) Создаём псевдоним модуля
sys.modules["TTS"] = importlib.import_module("TTS_my.TTS")
sys.modules["TTS.tts"] = importlib.import_module("TTS_my.TTS.tts")
sys.modules["TTS.tts.configs"] = importlib.import_module("TTS_my.TTS.tts.configs")
sys.modules["TTS.tts.configs.xtts_config"] = importlib.import_module("TTS_my.TTS.tts.configs.xtts_config")
sys.modules["TTS.tts.models"] = importlib.import_module("TTS_my.TTS.tts.models")
sys.modules["TTS.tts.models.xtts"] = importlib.import_module("TTS_my.TTS.tts.models.xtts")
sys.modules["TTS.utils"] = importlib.import_module("TTS_my.TTS.utils")
sys.modules["TTS.utils.io"] = importlib.import_module("TTS_my.TTS.utils.io")

import torch
from trainer import Trainer, TrainerArgs
from TTS_my.TTS.utils.audio import AudioProcessor

from datasets.preprocess import load_wav_feat_spk_data
from configs.gpt_hifigan_config import GPTHifiganConfig
from models.gpt_gan import GPTGAN


class GPTHifiganTrainer:
    def __init__(self, config):
        self.config = config
        # init audio processor
        self.ap = AudioProcessor(**config.audio.to_dict())
        # load training samples
        self.eval_samples, self.train_samples = load_wav_feat_spk_data(config.data_path, config.mel_path, config.spk_path, config.eval_split_size)
        self.model = GPTGAN(config, self.ap)

        if config.pretrain_path is not None:
            state_dict = torch.load(config.pretrain_path)
            hifigan_state_dict = {k.replace("xtts.hifigan_decoder.waveform_decoder.", "").replace("hifigan_decoder.waveform_decoder.", ""): v for k, v in state_dict["model"].items() if "hifigan_decoder" in k and "speaker_encoder" not in k}
            self.model.model_g.load_state_dict(hifigan_state_dict, strict=False)

            if config.train_spk_encoder:
                speaker_encoder_state_dict = {k.replace("xtts.hifigan_decoder.speaker_encoder.", "").replace("hifigan_decoder.waveform_decoder.", ""): v for k, v in state_dict["model"].items() if "hifigan_decoder" in k and "speaker_encoder" in k}
                self.model.speaker_encoder.load_state_dict(speaker_encoder_state_dict, strict=True)

    def train(self):
        # init the trainer and 🚀
        trainer = Trainer(
            TrainerArgs(), config, config.output_path, model=self.model, train_samples=self.train_samples, eval_samples=self.eval_samples
        )
        trainer.fit()


if __name__ == "__main__":
    config = GPTHifiganConfig(
        batch_size=100,
        eval_batch_size=2,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=5,
        epochs=1000,
        seq_len=8192,
        output_sample_rate=24000,
        gpt_latent_dim = 1024,
        pad_short=2000,
        use_noise_augment=False,
        eval_split_size=10,
        print_step=25,
        print_eval=False,
        mixed_precision=False,
        lr_gen=1e-4,
        lr_disc=1e-4,
        use_stft_loss=True,
        use_l1_spec_loss=True,
        data_path="ELEVEN_HIFI/wavs",
        mel_path="ELEVEN_HIFI/gpt_latents",
        spk_path ="ELEVEN_HIFI/speaker_embeddings",
        output_path="outputs",
        pretrain_path="/home/xtts_v2_training/src/run/training/XTTS_ELEVEN_ONLY_VALID_ALL-August-22-2025_08+11AM-0cbe12f/best_model_58710.pth",
        train_spk_encoder=False,
    )

    hifigan_trainer = GPTHifiganTrainer(config=config)
    hifigan_trainer.train()
