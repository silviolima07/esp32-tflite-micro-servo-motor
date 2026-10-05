idf_component_register(
    SRCS
        "main.cc"
        "model/model.cc"

        "audio/teste_abrir_01.cc"
        "audio/teste_abrir_02.cc"
        "audio/teste_fechar_01.cc"
        "audio/teste_fechar_02.cc"
        "audio/audio_preprocessing.cc"
        "audio/mfcc.cc"
        "audio/teste_05.cc"
        "audio/teste_06.cc"

        "servo_motor.cc"
        "audio/validacao_fechar_01.cc"

    INCLUDE_DIRS
        "."
        "model"
        "audio"

    REQUIRES
        esp-tflite-micro
        esp_driver_ledc
        esp_driver_usb_serial_jtag
)
