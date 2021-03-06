"""Hyper parameters."""
__author__ = 'Erdene-Ochir Tuguldur'


class HParams:
    """Hyper parameters"""

    disable_progress_bar = False  # set True if you don't want the progress bar in the console

    logdir = "logdir"  # log dir where the checkpoints and tensorboard files are saved
    data_path = '/media/DATA/SWARA_DATA/SWARA_wav_16k_new_trim/'
    mels_path = '/media/DATA/SWARA_DATA/SWARA_wav_16k_new_trim/'
    mags_path = '/media/DATA/SWARA_DATA/SWARA_wav_16k_new_trim/'

    # audio.py options, these values are from https://github.com/Kyubyong/dc_tts/blob/master/hyperparams.py
    reduction_rate = 4  # melspectrogram reduction rate, don't change because SSRN is using this rate
    n_fft = 2048 # fft points (samples)
    n_mels = 80  # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20
    # sr = 22050  # Sampling rate
    sr = 16000  # Sampling rate
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    max_N = 180  # Maximum number of characters.
    max_T = 210  # Maximum number of mel frames.

    e = 128  # embedding dimension
    d = 256  # Text2Mel hidden unit dimension
    c = 512+128  # SSRN hidden unit dimension

    dropout_rate = 0.05  # dropout

    # Text2Mel network options
    text2mel_lr = 0.005  # learning rate
    text2mel_max_iteration = 800000  # max train step
    text2mel_weight_init = 'none'  # 'kaiming', 'xavier' or 'none'
    text2mel_normalization = 'layer'  # 'layer', 'weight' or 'none'
    text2mel_basic_block = 'highway'  # 'highway', 'gated_conv' or 'residual'

    # SSRN network options
    ssrn_lr = 0.0005  # learning rate
    ssrn_max_iteration = 150000  # max train step
    ssrn_weight_init = 'kaiming'  # 'kaiming', 'xavier' or 'none'
    ssrn_normalization = 'weight'  # 'layer', 'weight' or 'none'
    ssrn_basic_block = 'residual'  # 'highway', 'gated_conv' or 'residual'

    # multispeaker params
    # possible values: text_encoder_input, text_encoder_towards_end, audio_decoder_input, ssrn_input, audio_encoder_input
    # multispeaker = []
    # multispeaker = ['learn_channel_contributions']
    # multispeaker = ['text_encoder_input', 'text_encoder_towards_end', 'audio_encoder_input', 'audio_decoder_input']
    # multispeaker = ['text_encoder_input', 'audio_encoder_input', 'audio_decoder_input']
    multispeaker = ['audio_encoder_input', 'audio_decoder_input']
    speaker_list = ['<PADDING>'] + ['BAS', 'BEA', 'CAU', 'DCS', 'DDM', 'EME', 'FDS', 'HTM', 'IPS', 'MARIA', 'PCS',
                                    'PMM', 'PSS', 'RMS', 'SAM', 'SDS', 'SGS', 'TIM', 'TSS']
    nspeakers = len(speaker_list) + 10
    speaker_embedding_size = 128

    speaker2ix = dict(zip(speaker_list, range(len(speaker_list))))

    # added for speaker recognition
    use_additional_speaker_loss = 1
    speaker_loss_type = 'Cosine_Similarity'  # 'Cosine_Similarity', 'Equal_Error_Rate'
    speaker_recognition_model_path = 'pretrained_voxceleb_model/model_checkpoints/model000000500.model'
    embedding_files = '/home/sintero-gpu1/work/Speaker-Recognition/voxceleb_trainer/embeddings_new/'
    test_file_list = 'meta/swara_list.txt'
    test_path = '/media/DATA/SWARA_DATA/SWARA_wav_16k_new_trim/'

    same_spk_samples = 12
    other_spk_samples = 1

    # temp settings
    Y_save_path = 'Y_files/'
