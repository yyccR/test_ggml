
#include <iostream>
#include "whisper.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

bool read_wav(const std::string & fname, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s, bool stereo) {
    drwav wav;
    std::vector<uint8_t> wav_data; // used for pipe input from stdin

    if (fname == "-") {
        {
            uint8_t buf[1024];
            while (true)
            {
                const size_t n = fread(buf, 1, sizeof(buf), stdin);
                if (n == 0) {
                    break;
                }
                wav_data.insert(wav_data.end(), buf, buf + n);
            }
        }

        if (drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr) == false) {
            fprintf(stderr, "error: failed to open WAV file from stdin\n");
            return false;
        }

        fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, wav_data.size());
    }
    else if (drwav_init_file(&wav, fname.c_str(), nullptr) == false) {
        fprintf(stderr, "error: failed to open '%s' as WAV file\n", fname.c_str());
        return false;
    }

    if (wav.channels != 1 && wav.channels != 2) {
        fprintf(stderr, "%s: WAV file '%s' must be mono or stereo\n", __func__, fname.c_str());
        return false;
    }

    if (stereo && wav.channels != 2) {
        fprintf(stderr, "%s: WAV file '%s' must be stereo for diarization\n", __func__, fname.c_str());
        return false;
    }

//    if (wav.sampleRate != COMMON_SAMPLE_RATE) {
//        fprintf(stderr, "%s: WAV file '%s' must be %i kHz\n", __func__, fname.c_str(), COMMON_SAMPLE_RATE/1000);
//        return false;
//    }

    if (wav.bitsPerSample != 16) {
        fprintf(stderr, "%s: WAV file '%s' must be 16-bit\n", __func__, fname.c_str());
        return false;
    }

    const uint64_t n = wav_data.empty() ? wav.totalPCMFrameCount : wav_data.size()/(wav.channels*wav.bitsPerSample/8);

    std::vector<int16_t> pcm16;
    pcm16.resize(n*wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);

    // convert to mono, float
    pcmf32.resize(n);
    if (wav.channels == 1) {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[i])/32768.0f;
        }
    } else {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
        }
    }

    if (stereo) {
        // convert to stereo, float
        pcmf32s.resize(2);

        pcmf32s[0].resize(n);
        pcmf32s[1].resize(n);
        for (uint64_t i = 0; i < n; i++) {
            pcmf32s[0][i] = float(pcm16[2*i])/32768.0f;
            pcmf32s[1][i] = float(pcm16[2*i + 1])/32768.0f;
        }
    }

    return true;
}

void test_whisper_ggml_infer() {
    std::string model_file = "/Users/yang/CLionProjects/test_ggml/whisper/ggml-whisper-tiny.bin";
    std::string wav_file = "/Users/yang/CLionProjects/test_ggml/data/audio/60351dbf545f99407d4d71ef_2.wav";

    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM
    read_wav(wav_file, pcmf32, pcmf32s, false);

    const char *sysinfo = whisper_print_system_info();
    std::cout << "whisper_print_system_info(): " << sysinfo << std::endl;
    // 加载音频
//    std::vector<float> pcmf32;
//    drwav wav;
//    size_t audio_dataSize=0;
//    char* audio_buffer = nullptr;
//    drwav_init_file(&wav, wav_file.c_str(), NULL);
//    std::cout << "read wav file: " << wav_file << std::endl;
//    std::cout << "wav file channels: " << wav.channels << ", sample rate: " << wav.sampleRate << ", totalPCMFrameCount: " << wav.totalPCMFrameCount << std::endl;
//    int n = wav.totalPCMFrameCount;
//    std::vector<int16_t> pcm16;
//    pcm16.resize(n*wav.channels);
//    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
//    drwav_uninit(&wav);
//    // convert to mono, float
//    pcmf32.resize(n);
//    if (wav.channels == 1) {
//        for (int i = 0; i < n; i++) {
//            pcmf32[i] = float(pcm16[i])/32768.0f;
////            std::cout << "pcmf32[i]" << pcmf32[i] << std::endl;
//        }
//    } else {
//        for (int i = 0; i < n; i++) {
//            pcmf32[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
//        }
//    }

    //Hack if the audio file size is less than 30ms append with 0's
//    pcmf32.resize((WHISPER_SAMPLE_RATE*WHISPER_CHUNK_SIZE),0);


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