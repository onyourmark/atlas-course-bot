from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import main
from knowledge import build_course_chunks

@pytest.fixture
def student_chat(monkeypatch):
    monkeypatch.setattr(main, 'COURSES', {'personal-test': {'_managed': True, '_status': 'published', '_owner_id': 'faculty', '_provider': 'anthropic', '_model': 'course-model'}})
    monkeypatch.setattr(main, 'SYSTEM_PROMPTS', {'personal-test': 'Use the course materials. Refer to the instructor.'})
    monkeypatch.setattr(main, 'COURSE_SOURCE_CHUNKS', {'personal-test': build_course_chunks('', {'lecture.txt': 'A tool is a callable function that performs an action.'})})
    store = MagicMock()
    store.remaining_questions.return_value = 0
    monkeypatch.setattr(main, '_require_pilot_store', lambda: store)
    return TestClient(main.app), store

@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
def test_personal_key_separate_from_faculty(student_chat, provider):
    client, store = student_chat
    key = 'student-secret-not-for-storage'
    constructor = MagicMock()
    model_client = constructor.return_value.__enter__.return_value
    module = main.openai if provider == 'openai' else main.anthropic
    name = 'OpenAI' if provider == 'openai' else 'Anthropic'
    with patch.object(module, name, constructor), patch.object(main, '_call_model', return_value=main.ProviderResponse('A callable function.', 20, 5)) as call, patch.object(main, '_get_client') as faculty:
        response = client.post('/course/personal-test/chat', headers={'X-ATLAS-Student-Key': key}, json={'message': 'What is a tool?', 'student_provider': provider, 'student_model': 'my-model'})
    assert response.status_code == 200, response.text
    assert response.json()['sources']
    assert constructor.call_args.kwargs['api_key'] == key
    assert constructor.call_args.kwargs['base_url'].startswith('https://api.')
    assert call.call_args.args[:3] == (provider, model_client, 'my-model')
    assert 'lecture.txt' in str(call.call_args)
    faculty.assert_not_called()
    store.record_usage.assert_not_called()
    store.decrypted_api_key.assert_not_called()
    assert key not in response.text


def test_default_still_enforces_course_allowance(student_chat):
    client, _ = student_chat
    with patch.object(main, '_call_course_model') as call:
        r = client.post('/course/personal-test/chat', json={'message': 'What is a tool?'})
    assert r.status_code == 429
    call.assert_not_called()


def test_local_prepares_same_sources_without_provider_or_quota(student_chat):
    client, store = student_chat
    with patch.object(main, '_get_client') as call:
        r = client.post('/course/personal-test/chat', json={'message': 'What is a tool?', 'student_provider': 'local', 'student_model': 'local-model:latest'})
    assert r.status_code == 200, r.text
    assert r.headers['cache-control'] == 'no-store'
    assert r.json()['local_request']['model'] == 'local-model:latest'
    assert 'callable function' in r.json()['local_request']['messages'][-1]['content']
    assert r.json()['sources']
    call.assert_not_called()
    store.record_usage.assert_not_called()


def test_missing_credentials_never_fall_back(student_chat):
    client, _ = student_chat
    with patch.object(main, '_get_client') as call:
        for payload in [dict(student_provider='openai', student_model='test'), dict(student_provider='local')]:
            r = client.post('/course/personal-test/chat', json={'message': 'What is a tool?', **payload})
            assert r.status_code == 400
    call.assert_not_called()


def test_provider_failure_does_not_echo_secret_or_use_faculty(student_chat):
    import httpx
    client, store = student_chat
    secret = 'student-secret-12345'
    error = main.openai.AuthenticationError('bad '+secret, response=httpx.Response(401, request=httpx.Request('POST', 'https://api.openai.com')), body=None)
    with patch.object(main.openai, 'OpenAI'), patch.object(main, '_call_model', side_effect=error), patch.object(main, '_get_client') as faculty:
        r = client.post('/course/personal-test/chat', headers={'X-ATLAS-Student-Key': secret}, json={'message': 'What is a tool?', 'student_provider': 'openai', 'student_model': 'test'})
    assert r.status_code == 502
    assert secret not in r.text
    faculty.assert_not_called()
    store.record_usage.assert_not_called()


def test_no_sources_does_not_prepare_local_call(student_chat):
    client, _ = student_chat
    r = client.post('/course/personal-test/chat', json={'message': 'Explain photosynthesis', 'student_provider': 'local', 'student_model': 'test'})
    assert r.status_code == 200
    assert 'local_request' not in r.json()
    assert not r.json()['materials_found']
