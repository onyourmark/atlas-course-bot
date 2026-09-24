import time
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
import main
import student_credentials
from student_credentials import StudentCredentials
from pilot_platform import PilotStore, generate_encryption_key
from knowledge import build_course_chunks

@pytest.fixture
def vault_app(tmp_path, monkeypatch):
    store = PilotStore(tmp_path, generate_encryption_key()); store.initialize()
    monkeypatch.setattr(main, '_require_pilot_store', lambda: store)
    monkeypatch.setattr(main, 'SECURE_COOKIES', True)
    monkeypatch.setattr(main, 'COURSES', {'one': {}, 'two': {}})
    monkeypatch.setattr(main, 'SYSTEM_PROMPTS', {'one':'Use course sources.', 'two':'Use course sources.'})
    monkeypatch.setattr(main, 'COURSE_SOURCE_CHUNKS', {'one':build_course_chunks('', {'lecture.txt':'A tool is a callable function.'})})
    return store, TestClient(main.app, base_url='https://testserver')

def save(client, weeks=1, **kwargs):
    return client.put('/course/one/student-key', headers={'X-ATLAS-Settings':'1','X-ATLAS-Student-Key':'student-private-key'},
                      json={'student_provider':'openai','student_model':'my-model','weeks':weeks, **kwargs})

@pytest.mark.parametrize('weeks', [1,2,3,4])
def test_saved_key_encryption_restore_and_cookie(vault_app, weeks):
    store, client = vault_app
    before = time.time()
    response = save(client, weeks)
    assert response.status_code == 200, response.text
    assert before + weeks*7*86400 <= response.json()['expires_at'] <= time.time() + weeks*7*86400
    cookie=response.headers['set-cookie']
    assert 'HttpOnly' in cookie and 'Secure' in cookie and 'SameSite=strict' in cookie and 'Path=/course/one' in cookie
    assert 'student-private-key' not in response.text
    data = client.get('/course/one/student-key').json()
    assert data['saved'] and data['provider']=='openai' and 'key' not in data
    with store._connect() as db:
        row=dict(db.execute('SELECT * FROM student_credentials').fetchone())
    assert 'student-private-key' not in str(row)
    token=client.cookies.get('atlas_student_key')
    assert token not in str(row)
    assert StudentCredentials(store).get(token,'one')['key']=='student-private-key'
    assert client.get('/course/two/student-key').json()=={'saved':False}
    stranger=TestClient(main.app,base_url='https://testserver')
    assert stranger.get('/course/one/student-key').json()=={'saved':False}


def test_saved_key_used_without_returning_it_to_browser(vault_app):
    _, client=vault_app;save(client)
    with patch.object(main.openai,'OpenAI') as ctor, patch.object(main,'_call_model',return_value=main.ProviderResponse('A function.',1,2)), patch.object(main,'_get_client') as faculty:
        r=client.post('/course/one/chat',json={'message':'What is a tool?','student_provider':'openai','student_model':'my-model','use_saved_key':True})
    assert r.status_code==200,r.text
    assert ctor.call_args.kwargs['api_key']=='student-private-key'
    faculty.assert_not_called()
    assert 'student-private-key' not in r.text


def test_expiry_and_forgetting_revoke_use(vault_app,monkeypatch):
    store,client=vault_app; result=save(client)
    with monkeypatch.context() as m:
        m.setattr(student_credentials.time,'time',lambda:result.json()['expires_at']+1)
        assert client.get('/course/one/student-key').json()=={'saved':False}
        with store._connect() as db: assert db.execute('SELECT count(*) FROM student_credentials').fetchone()[0]==0
        r=client.post('/course/one/chat',json={'message':'What is a tool?','student_provider':'openai','student_model':'my-model','use_saved_key':True})
        assert r.status_code==401
    save(client); old=client.cookies.get('atlas_student_key')
    assert client.delete('/course/one/student-key',headers={'X-ATLAS-Settings':'1'}).status_code==200
    assert StudentCredentials(store).get(old,'one') is None
    assert client.get('/course/one/student-key').json()=={'saved':False}


def test_renewal_rotation_and_provider_isolation(vault_app):
    store,client=vault_app;save(client);old=client.cookies.get('atlas_student_key')
    r=client.put('/course/one/student-key',headers={'X-ATLAS-Settings':'1'},json={'student_provider':'openai','student_model':'another','weeks':4,'use_saved_key':True})
    assert r.status_code==200,r.text
    assert StudentCredentials(store).get(old,'one') is None
    r=client.post('/course/one/chat',json={'message':'What is a tool?','student_provider':'deepseek_cn','student_model':'deepseek-flash','use_saved_key':True})
    assert r.status_code==400


def test_settings_csrf_and_duration_validation(vault_app):
    _,client=vault_app
    body={'student_provider':'openai','student_model':'test','weeks':1}
    assert client.put('/course/one/student-key',json=body).status_code==403
    assert client.put('/course/one/student-key',json=body,headers={'X-ATLAS-Settings':'1','Origin':'https://evil.test'}).status_code==403
    for weeks in [0,5,-1]: assert save(client,weeks).status_code==422


def test_local_token_restore_and_expiry(vault_app):
    _,client=vault_app
    r=save(client,student_provider='local',student_model='local-model',local_url='http://localhost:11434/v1')
    assert r.status_code==200,r.text
    assert client.get('/course/one/student-key').json()['key']=='student-private-key'
    assert save(client,student_provider='local',local_url='https://evil.test').status_code==400
