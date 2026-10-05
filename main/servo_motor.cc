#include "servo_motor.h"

#include <stdio.h>
#include "driver/ledc.h"

static constexpr int SERVO_GPIO = 5;

static constexpr int SERVO_MIN_US = 500;
static constexpr int SERVO_MAX_US = 2500;
static constexpr int SERVO_PERIOD_US = 20000; // 50 Hz

static constexpr ledc_mode_t SERVO_SPEED_MODE = LEDC_LOW_SPEED_MODE;
static constexpr ledc_timer_t SERVO_TIMER = LEDC_TIMER_0;
static constexpr ledc_channel_t SERVO_CHANNEL = LEDC_CHANNEL_0;
static constexpr ledc_timer_bit_t SERVO_RESOLUTION = LEDC_TIMER_14_BIT;

void servo_init()
{
    ledc_timer_config_t timer_config = {};
    timer_config.speed_mode = SERVO_SPEED_MODE;
    timer_config.timer_num = SERVO_TIMER;
    timer_config.duty_resolution = SERVO_RESOLUTION;
    timer_config.freq_hz = 50;
    timer_config.clk_cfg = LEDC_AUTO_CLK;

    ledc_timer_config(&timer_config);

    ledc_channel_config_t channel_config = {};
    channel_config.gpio_num = SERVO_GPIO;
    channel_config.speed_mode = SERVO_SPEED_MODE;
    channel_config.channel = SERVO_CHANNEL;
    channel_config.intr_type = LEDC_INTR_DISABLE;
    channel_config.timer_sel = SERVO_TIMER;
    channel_config.duty = 0;
    channel_config.hpoint = 0;

    ledc_channel_config(&channel_config);

    printf("Servo inicializado no GPIO %d\n", SERVO_GPIO);
}

void servo_set_angle(int angle)
{
    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;

    int pulse_us =
        SERVO_MIN_US +
        (angle * (SERVO_MAX_US - SERVO_MIN_US)) / 180;

    const uint32_t max_duty = (1 << 14) - 1;

    uint32_t duty =
        (pulse_us * max_duty) / SERVO_PERIOD_US;

    ledc_set_duty(
        SERVO_SPEED_MODE,
        SERVO_CHANNEL,
        duty
    );

    ledc_update_duty(
        SERVO_SPEED_MODE,
        SERVO_CHANNEL
    );

    printf("Servo -> %d graus\n", angle);
}

void abrir_porta()
{
    printf("Acao: ABRINDO PORTA\n");
    servo_set_angle(0);
}

void fechar_porta()
{
    printf("Acao: FECHANDO PORTA\n");
    servo_set_angle(90);
}