
#include <iostream>
#include "whisper.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

void test_whisper_ggml_infer() {
    std::string model_file = "/Users/yang/CLionProjects/test_ggml/whisper/ggml-whisper-tiny.bin";
    std::string wav_file = "/Users/yang/CLionProjects/test_ggml/data/audio/60351dbf545f99407d4d71ef_2.wav";
    // 加载音频
    std::vector<float> pcmf32;
    drwav wav;
    size_t audio_dataSize=0;
    char* audio_buffer = nullptr;
    drwav_init_file(&wav, wav_file.c_str(), NULL);
    std::cout << "read wav file: " << wav_file << std::endl;
    std::cout << "wav file channels: " << wav.channels << ", sample rate: " << wav.sampleRate << ", totalPCMFrameCount: " << wav.totalPCMFrameCount << std::endl;
    int n = wav.totalPCMFrameCount;
    std::vector<int16_t> pcm16;
    pcm16.resize(n*wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);
    // convert to mono, float
    pcmf32.resize(n);
    if (wav.channels == 1) {
        for (int i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[i])/32768.0f;
//            std::cout << "pcmf32[i]" << pcmf32[i] << std::endl;
        }
    } else {
        for (int i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
        }
    }

    //Hack if the audio file size is less than 30ms append with 0's
    pcmf32.resize((WHISPER_SAMPLE_RATE*WHISPER_CHUNK_SIZE),0);

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.translate = false;
    params.language = "zh";
    params.offset_ms = 0;
    params.no_context = true;
    params.single_segment = false;

    struct whisper_context * ctx = whisper_init_from_file(model_file.c_str());
    if (whisper_full(ctx, params, pcmf32.data(), pcmf32.size()) != 0) {
        fprintf(stderr, "failed to process audio\n");
        return ;
    }

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        printf("%s", text);
    }

    whisper_free(ctx);
}