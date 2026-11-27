"""Sample verses for the demo.

Each entry is a real text with a note about what it demonstrates.  Keeping the
provenance in the UI matters: a heritage tool that cannot say where its examples
came from is asking to be trusted rather than to be checked.
"""

SAMPLE_VERSES = [
    {
        "id": "agnimile",
        "title": "Agniṃ īḷe — RV 1.1.1, in Sāma rendition",
        "sanskrit": "हाउ अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् ।\nहोता॑रं रत्न॒धात॑मम् ॥",
        "iast": "hāu agnimīḷe purohitaṃ yajñasya devam ṛtvijam | hotāraṃ ratnadhātamam ||",
        "mode_hint": "samhita",
        "note": (
            "The opening ṛc of the Ṛgveda, which the Sāmaveda also carries. The "
            "accents in the text drive the notes directly — this is recitation "
            "(pāṭha), so the mode is saṃhitā, not gāna."
        ),
        "source": "Ṛgveda 1.1.1 / Sāmaveda Kauthuma",
    },
    {
        "id": "gana_agnimile",
        "title": "The same ṛc as a gāna text (with svara numerals)",
        "sanskrit": "हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विज॑म्",
        "iast": "hāu 4 agnimīḷe 3 purohitaṃ 2 yajñasya 1 devamṛtvijam",
        "mode_hint": "gana",
        "note": (
            "The same words with sāma notation: a stobha (hāu) opens the phrase and "
            "each syllable carries a svara number. In gāna mode nothing is invented — "
            "the notes are read out of the text."
        ),
        "source": "Grāma-geya gāna (notation form)",
    },
    {
        "id": "savitri",
        "title": "Sāvitrī (Gāyatrī mantra) — the sāma setting",
        "sanskrit": "ॐ भूर्भुव॒: स्व॒: तत्स॑वि॒तुर्वरे᳚ण्यं॒ भर्गो॑ दे॒वस्य॑ धीमहि ।\nधियो॒ यो न॑: प्रचो॒दया᳚त् ॥",
        "iast": "oṃ bhūrbhuvaḥ svaḥ tatsaviturvareṇyaṃ bhargo devasya dhīmahi | dhiyo yo naḥ pracodayāt ||",
        "mode_hint": "samhita",
        "note": (
            "The most-recited Vedic verse, and the one most people have heard set to "
            "a tune. The sāma setting is not the same melody as the popular "
            "instrumental versions. Soft retroflexes and the visarga allophones are "
            "audible here."
        ),
        "source": "Ṛgveda 3.62.10",
    },
    {
        "id": "aranyaka",
        "title": "Āraṇyaka-gāna register (forest song-book)",
        "sanskrit": "होवा आ॒यु॒र्दा दे॒वो अ॒ग्निर्वैश्वा॑न॒रो वे॑द गृ॒हो भू॑मिं स्पृ॒शन्",
        "iast": "hovā āyurdā devo agnir vaiśvānaro veda gṛho bhūmiṃ spṛśan",
        "mode_hint": "style",
        "note": (
            "Rendered on the Āraṇyaka skeleton: a fourth lower than Grāma-geya, with "
            "longer holds and more mandra. Switch the model to hear the register "
            "difference on the same words."
        ),
        "source": "Āraṇyaka-gāna",
    },
    {
        "id": "pavamanam",
        "title": "Pavamāna — the Soma hymn of the Sāmaveda",
        "sanskrit": "हाउ प॒व॒स्व सोम॑ मधु॒मत्त॑मो म॒दिन्त॑मः ।\nप॒वस्व सोम॑ मधु॒मान् ॥",
        "iast": "hāu pavasva soma madhumattamo madintamaḥ | pavasva soma madhumān ||",
        "mode_hint": "style",
        "note": (
            "The Pavamāna verses are the heart of the Sāmaveda and the bulk of the "
            "Grāma-geya book. This is the 'style' mode on the Grāma-geya skeleton: "
            "the documented vikāras applied, the right musical grammar, but not the "
            "specific sāman a śākhā would sing."
        ),
        "source": "Sāmaveda Pūrvārcika (Pavamāna)",
    },
    {
        "id": "nama",
        "title": "A plain Sanskrit verse (no Vedic accents)",
        "sanskrit": "नमस्ते वायो त्वमेव प्रत्यक्षं ब्रह्मासि त्वमेव प्रत्यक्षं ब्रह्म वदिष्यामि ऋतं वदिष्यामि",
        "iast": "namaste vāyo tvameva pratyakṣaṃ brahmāsi tvameva pratyakṣaṃ brahma vadiṣyāmi ṛtaṃ vadiṣyāmi",
        "mode_hint": "style",
        "note": (
            "Try pasting anything Sanskrit here: the frontend is not a Sanskrit-"
            "only path, and the result is honestly labelled 'styled'. This is the "
            "case where a system that claimed to know the right tune would be lying."
        ),
        "source": "Śukla Yajurveda (shown as a non-Vedic-notation example)",
    },
    {
        "id": "kannada",
        "title": "Any Indic script — Kannada input",
        "sanskrit": "ಹಾಉ ಅಗ್ನಿಮೀಳೇ ಪುರೋಹಿತಂ ಯಜ್ಞಸ್ಯ ದೇವಮೃತ್ವಿಜಮ್",
        "iast": "hāu agnimīḷe purohitaṃ yajñasya devamṛtvijam",
        "mode_hint": "auto",
        "note": (
            "Script is detected and converted to Devanāgarī internally. The pitfall "
            "this avoids: Hindi word-processors delete the final 'a' (rāma -> rām), "
            "which would make nonsense of Vedic text. The frontend never applies "
            "schwa deletion."
        ),
        "source": "Transliterated from RV 1.1.1",
    },
]
