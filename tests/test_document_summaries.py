from knowledge import build_course_chunks, search_chunk_matches

def test_chapter_words_numbers_and_filename_use_full_chapter():
    chapter = "Opening: model calls. " + "Parsing and validation. " * 200 + "Ending: retry budgets and stopping."
    chunks = build_course_chunks("", {
        "week2-transcript.txt": "Chapter two begins here. Chapter two is the code around a model call.",
        "chapter_02_revised.md": chapter,
        "chapter_03_revised.md": "Unrelated material.",
        "Chapter_2_Read_Aloud.pptx": "Short classroom adaptation.",
    })
    for question in ("What are the main points in chapter two?",
                     "Summarize chapter 2", "What are the main points in chapter_02_revised.md ?"):
        result = search_chunk_matches(question, chunks, max_chunks=3)
        assert len(result) == 1
        assert result[0]["source"] == "chapter_02_revised.md"
        assert "Opening: model calls." in result[0]["text"]
        assert "Ending: retry budgets and stopping." in result[0]["text"]
    assert not search_chunk_matches("Summarize chapter four", chunks)

def test_explicit_slide_filename_is_respected():
    chunks = build_course_chunks("", {
        "chapter_02_revised.md": "Book text",
        "Chapter_2_Read_Aloud.pptx": "Slide contents",
    })
    result = search_chunk_matches("Summarize Chapter_2_Read_Aloud.pptx", chunks)
    assert result[0]["source"] == "Chapter_2_Read_Aloud.pptx"
