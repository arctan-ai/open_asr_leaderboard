"""Static ASR language capabilities exposed by the evaluation console.

The provider documentation URLs below are intentionally kept beside the data so
language changes can be reviewed without adding runtime calls to provider APIs.
"""

from __future__ import annotations


AUTO = {"code": "unknown", "label": "Automatic detection"}
MULTI = {"code": "multi", "label": "Multilingual / code-switching"}

LANGUAGE_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fo": "Faroese", "fr": "French", "gl": "Galician", "gu": "Gujarati",
    "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "la": "Latin", "lb": "Luxembourgish",
    "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian",
    "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Burmese", "nb": "Norwegian Bokmal", "ne": "Nepali", "nl": "Dutch",
    "nn": "Norwegian Nynorsk", "no": "Norwegian", "oc": "Occitan", "or": "Odia",
    "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "sd": "Sindhi",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sn": "Shona",
    "so": "Somali", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog",
    "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu",
    "uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba",
    "yue": "Cantonese", "zh": "Chinese",
    "ar-AE": "Arabic (UAE)", "ar-SA": "Arabic (Saudi Arabia)",
    "ar-QA": "Arabic (Qatar)", "ar-KW": "Arabic (Kuwait)",
    "ar-SY": "Arabic (Syria)", "ar-LB": "Arabic (Lebanon)",
    "ar-PS": "Arabic (Palestine)", "ar-JO": "Arabic (Jordan)",
    "ar-EG": "Arabic (Egypt)", "ar-SD": "Arabic (Sudan)",
    "ar-TD": "Arabic (Chad)", "ar-MA": "Arabic (Morocco)",
    "ar-DZ": "Arabic (Algeria)", "ar-TN": "Arabic (Tunisia)",
    "ar-IQ": "Arabic (Iraq)", "ar-IR": "Arabic (Iran)",
    "da-DK": "Danish (Denmark)", "de-CH": "German (Switzerland)",
    "en-AU": "English (Australia)", "en-GB": "English (United Kingdom)",
    "en-IN": "English (India)", "en-NZ": "English (New Zealand)",
    "en-US": "English (United States)", "es-419": "Spanish (Latin America)",
    "fr-CA": "French (Canada)", "gu-IN": "Gujarati (India)",
    "ko-KR": "Korean (South Korea)", "nl-BE": "Flemish",
    "pt-BR": "Portuguese (Brazil)", "pt-PT": "Portuguese (Portugal)",
    "sv-SE": "Swedish (Sweden)", "th-TH": "Thai (Thailand)",
    "zh-CN": "Chinese (Simplified)", "zh-HK": "Cantonese (Traditional)",
    "zh-Hans": "Chinese (Simplified script)", "zh-Hant": "Chinese (Traditional script)",
    "zh-TW": "Chinese (Traditional)",
}


def options(codes: list[str], *special: dict[str, str]) -> list[dict[str, str]]:
    return [*special, *({"code": code, "label": LANGUAGE_NAMES[code]} for code in codes)]


# https://www.assemblyai.com/docs/pre-recorded-audio/supported-languages
ASSEMBLY_U3 = options(
    [
        "ar", "zh", "da", "nl", "en", "fi", "fr", "de", "he", "hi",
        "it", "ja", "no", "pt", "es", "sv", "tr", "vi",
    ],
    AUTO,
)

# https://docs.cartesia.ai/api-reference/stt/transcribe
CARTESIA_WHISPER_CODES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
]

# https://developers.deepgram.com/docs/models-languages-overview/
DEEPGRAM_NOVA3_CODES = [
    "ar", "ar-AE", "ar-SA", "ar-QA", "ar-KW", "ar-SY", "ar-LB", "ar-PS",
    "ar-JO", "ar-EG", "ar-SD", "ar-TD", "ar-MA", "ar-DZ", "ar-TN", "ar-IQ",
    "ar-IR", "be", "bn", "bs", "bg", "ca", "zh-HK", "zh", "zh-CN", "zh-Hans",
    "zh-TW", "zh-Hant", "hr", "cs", "da", "da-DK", "nl", "nl-BE", "en",
    "en-US", "en-AU", "en-GB", "en-IN", "en-NZ", "et", "fi", "fr", "fr-CA",
    "de", "de-CH", "el", "gu", "gu-IN", "he", "hi", "hu", "id", "it", "ja",
    "kn", "ko", "ko-KR", "lv", "lt", "mk", "ms", "mr", "no", "fa", "pl",
    "pt", "pt-BR", "pt-PT", "ro", "ru", "sr", "sk", "sl", "es", "es-419",
    "sv", "sv-SE", "tl", "ta", "te", "th", "th-TH", "tr", "uk", "ur", "vi",
]

# https://learn.microsoft.com/azure/ai-services/speech-service/mai-transcribe
MICROSOFT_MAI_CODES = [
    "ar", "as", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "et",
    "fi", "fr", "gu", "hi", "hu", "id", "it", "ja", "kn", "ko", "lt", "ml",
    "mr", "nb", "nl", "or", "pa", "pl", "pt", "ro", "ru", "sk", "sl", "sv",
    "ta", "te", "th", "tr", "uk", "vi", "zh",
]

# https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
SARVAM_CODES = [
    "en-IN", "hi-IN", "as-IN", "bn-IN", "ur-IN", "kn-IN", "ne-IN", "ml-IN",
    "kok-IN", "mr-IN", "ks-IN", "od-IN", "sd-IN", "pa-IN", "sa-IN", "ta-IN",
    "sat-IN", "te-IN", "mni-IN", "brx-IN", "gu-IN", "mai-IN", "doi-IN",
]
SARVAM_NAMES = {
    "en-IN": "English (India)", "hi-IN": "Hindi", "as-IN": "Assamese",
    "bn-IN": "Bengali", "ur-IN": "Urdu", "kn-IN": "Kannada", "ne-IN": "Nepali",
    "ml-IN": "Malayalam", "kok-IN": "Konkani", "mr-IN": "Marathi",
    "ks-IN": "Kashmiri", "od-IN": "Odia", "sd-IN": "Sindhi", "pa-IN": "Punjabi",
    "sa-IN": "Sanskrit", "ta-IN": "Tamil", "sat-IN": "Santali", "te-IN": "Telugu",
    "mni-IN": "Manipuri", "brx-IN": "Bodo", "gu-IN": "Gujarati",
    "mai-IN": "Maithili", "doi-IN": "Dogri",
}
SARVAM_OPTIONS = [AUTO, *({"code": code, "label": SARVAM_NAMES[code]} for code in SARVAM_CODES)]

# https://docs.smallest.ai/waves/v-4-0-0/model-cards/speech-to-text/pulse
SMALLEST_BATCH_CODES = [
    "en", "hi", "de", "es", "ru", "it", "fr", "nl", "pt", "uk", "pl", "cs",
    "sk", "lv", "et", "ro", "fi", "sv", "bg", "hu", "da", "lt", "mt", "zh",
    "ja", "ko",
]
SMALLEST_BATCH = [
    {"code": "multi-eu", "label": "Automatic detection (European)"},
    {"code": "multi-asian", "label": "Automatic detection (Asian)"},
    *options(SMALLEST_BATCH_CODES),
]

# https://soniox.com/docs/stt/concepts/supported-languages
SONIOX_CODES = [
    "af", "sq", "ar", "az", "eu", "be", "bn", "bs", "bg", "ca", "zh", "hr",
    "cs", "da", "nl", "en", "et", "fi", "fr", "gl", "de", "el", "gu", "he",
    "hi", "hu", "id", "it", "ja", "kn", "kk", "ko", "lv", "lt", "mk", "ms",
    "ml", "mr", "no", "fa", "pl", "pt", "pa", "ro", "ru", "sr", "sk", "sl",
    "es", "sw", "sv", "tl", "ta", "te", "th", "tr", "uk", "ur", "vi", "cy",
]


MODEL_LANGUAGES = {
    "assembly/universal-stt": {
        "batch": ASSEMBLY_U3,
        "streaming": [AUTO],
    },
    "cartesia/ink-whisper": {
        "batch": options(CARTESIA_WHISPER_CODES),
        "streaming": [],
    },
    "cartesia/ink-2": {
        "batch": [],
        "streaming": [AUTO],
    },
    "deepgram/nova-3": {
        "batch": options(DEEPGRAM_NOVA3_CODES, MULTI),
        "streaming": options(DEEPGRAM_NOVA3_CODES, MULTI),
    },
    "microsoft/azure-speech-05-2026": {
        "batch": options(MICROSOFT_MAI_CODES, AUTO),
        "streaming": [],
    },
    "microsoft/MAI-Transcribe-1.5": {
        "batch": [],
        "streaming": options(MICROSOFT_MAI_CODES, AUTO),
    },
    "sarvam/saaras:v3": {
        "batch": SARVAM_OPTIONS,
        "streaming": SARVAM_OPTIONS,
    },
    "smallestai/pulse": {
        "batch": SMALLEST_BATCH,
        "streaming": [],
    },
    "soniox/stt-async-v5": {
        "batch": options(SONIOX_CODES, AUTO),
        "streaming": options(SONIOX_CODES, AUTO),
    },
    "soniox/stt-rt-v5": {
        "batch": [],
        "streaming": options(SONIOX_CODES, AUTO),
    },
}


def effective_mode(model_name: str, streaming: bool) -> str:
    if model_name == "soniox/stt-rt-v5":
        return "streaming"
    return "streaming" if streaming else "batch"


def language_options(model_name: str, streaming: bool) -> list[dict[str, str]]:
    return MODEL_LANGUAGES.get(model_name, {}).get(
        effective_mode(model_name, streaming), []
    )
