// Keys are kept in memory or encrypted on ATLAS; browser storage never holds keys.
let studentProviderCatalog = null;
let studentRestorePromise = null, studentRestoreError = "";
let studentConnection = {provider: 'course', model: '', key: '', url: ''};
function studentKeyPath() { return '/course/' + encodeURIComponent(courseId) + '/student-key'; }
function updateStudentModelButton() {
    document.getElementById('model-settings-button').textContent = studentConnection.provider === 'course' ? 'Model: Course default' : 'Model: ' + studentConnection.model;
}
async function restoreStudentConnection() {
    try {
        const response = await fetch(studentKeyPath(), {cache: 'no-store', credentials: 'same-origin'});
        if (!response.ok) throw Error();
        const data = await response.json();
        if (data.saved) studentConnection = {...data, key: data.key || ''};
        studentRestoreError = '';
        updateStudentModelButton();
    } catch (_) { studentRestoreError = 'Could not check your saved model settings. Refresh the page before sending a question.'; }
}
async function deleteSavedStudentKey() {
    const response = await fetch(studentKeyPath(), {method: 'DELETE', headers: {'X-ATLAS-Settings': '1'}, credentials: 'same-origin'});
    if (!response.ok) throw Error('Could not delete the saved key. Please try again.');
}
async function forgetStudentKey() {
    try {
        await deleteSavedStudentKey();
        studentConnection = {provider: 'course', model: '', key: '', url: ''};
        document.getElementById('student-key').value = '';
        document.getElementById('student-provider').value = 'course';
        document.getElementById('student-remember').value = '0';
        updateStudentFields(); updateStudentModelButton();
        document.getElementById('student-saved-status').textContent = 'Saved key deleted. Using the course default.';
    } catch (error) { document.getElementById('student-settings-error').textContent = error.message; }
}
function localModelURL(value) {
    let url;
    try { url = new URL(value); } catch (_) { throw Error('Enter a local address such as http://localhost:11434/v1.'); }
    if (!['http:', 'https:'].includes(url.protocol) || !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) || url.username || url.password || url.search || url.hash) {
        throw Error('Use a localhost address on your own computer, without passwords or query parameters.');
    }
    return url.href.replace(/\/+$/, '') + '/chat/completions';
}
async function loadStudentProviderCatalog() {
    if (studentProviderCatalog) return;
    const response = await fetch('/static/student_model_providers.json?v=1', {cache: 'no-store'});
    if (!response.ok) throw Error('Could not load the provider options. Close and reopen Model settings.');
    const catalog = await response.json();
    const select = document.getElementById('student-provider');
    for (const [groupName, ids] of [
        ['China-based services', ['deepseek_cn', 'qwen_cn', 'kimi_cn', 'glm_cn', 'minimax_cn']],
        ['U.S.-hosted Chinese models', ['qwen_us', 'fireworks_us']]
    ]) {
        const group = document.createElement('optgroup'); group.label = groupName;
        ids.forEach(id => { const item = document.createElement('option'); item.value = id; item.textContent = catalog[id].label; group.append(item); });
        select.append(group);
    }
    studentProviderCatalog = catalog;
}
function chooseStudentModel() {
    const preset = document.getElementById('student-model-choice').value;
    document.getElementById('student-model-custom').hidden = preset !== '';
    document.getElementById('student-model').value = preset;
}
function updateStudentFields() {
    const provider = document.getElementById('student-provider').value;
    document.getElementById('student-custom-fields').hidden = provider === 'course';
    document.getElementById('student-local-fields').hidden = provider !== 'local';
    document.getElementById('student-key-label').textContent = provider === 'local' ? 'Local server token (optional)' : 'Your API key';
    document.getElementById('student-key').value = '';
    document.getElementById('student-key').placeholder = studentConnection.saved && provider === studentConnection.provider ? 'Saved key: leave blank to keep it' : '';
    document.getElementById('student-settings-error').textContent = '';
    const option = studentProviderCatalog?.[provider];
    const choices = document.getElementById('student-model-choice'); choices.replaceChildren();
    (option?.models || []).forEach(([id, name]) => { const item = document.createElement('option'); item.value = id; item.textContent = name; choices.append(item); });
    if (!option?.restricted_models) { const item = document.createElement('option'); item.value = ''; item.textContent = 'Enter a different model ID'; choices.append(item); }
    document.getElementById('student-model-presets').hidden = !option;
    document.getElementById('student-workspace-field').hidden = !option?.workspace;
    document.getElementById('student-workspace').value = '';
    const help = document.getElementById('student-provider-help'); help.textContent = option?.help || '';
    if (option) { const link = document.createElement('a'); link.href = option.docs; link.textContent = ' Provider setup'; link.target = '_blank'; link.rel = 'noopener noreferrer'; help.append(link); }
    chooseStudentModel();
}
async function openStudentSettings() {
    document.getElementById('student-model-dialog').showModal();
    const apply = document.getElementById('student-apply'); apply.disabled = true;
    try { if (studentRestorePromise) await studentRestorePromise; await loadStudentProviderCatalog(); }
    catch (_) { document.getElementById('student-settings-error').textContent = 'Could not load the provider options. Close and reopen Model settings.'; return; }
    finally { apply.disabled = false; }
    document.getElementById('student-provider').value = studentConnection.provider;
    updateStudentFields();
    const choices = document.getElementById('student-model-choice');
    choices.value = Array.from(choices.options).some(item => item.value === studentConnection.model) ? studentConnection.model : '';
    chooseStudentModel();
    document.getElementById('student-model').value = studentConnection.model;
    document.getElementById('student-workspace').value = studentConnection.workspace || '';
    document.getElementById('student-key').value = studentConnection.key;
    document.getElementById('student-url').value = studentConnection.url || 'http://localhost:11434/v1';
    document.getElementById('student-remember').value = String(studentConnection.weeks || 0);
    document.getElementById('student-saved-status').textContent = studentConnection.saved ? 'Saved in this browser until ' + new Date(studentConnection.expires_at * 1000).toLocaleString() + '.' : 'No key saved for this browser and course.';
}

async function saveStudentSettings(event) {
    event.preventDefault();
    const provider = document.getElementById('student-provider').value;
    const model = document.getElementById('student-model').value.trim();
    const key = document.getElementById('student-key').value.trim();
    const url = document.getElementById('student-url').value.trim();
    const workspace = document.getElementById('student-workspace').value.trim();
    const weeks = Number(document.getElementById('student-remember').value);
    const reuse = !!studentConnection.saved && studentConnection.provider === provider && !key;
    const apply = document.getElementById('student-apply'); apply.disabled = true;
    try {
        if (provider !== 'course' && !/^[A-Za-z0-9][A-Za-z0-9._:/-]{0,149}$/.test(model)) throw Error('Enter the exact model ID shown by your provider or local server.');
        if (provider !== 'course' && provider !== 'local' && key.length < 10 && !(reuse && weeks > 0)) throw Error('Enter your own API key.');
        if (studentProviderCatalog?.[provider]?.workspace && !/^[A-Za-z0-9][A-Za-z0-9-]{0,62}$/.test(workspace)) throw Error('Enter your Alibaba workspace ID, not its URL.');
        if (provider === 'local') localModelURL(url);
        if (provider !== 'course' && weeks > 0) {
            const headers = {'Content-Type': 'application/json', 'X-ATLAS-Settings': '1'};
            if (key) headers['X-ATLAS-Student-Key'] = key;
            const response = await fetch(studentKeyPath(), {method: 'PUT', credentials: 'same-origin', headers,
                body: JSON.stringify({student_provider: provider, student_model: model, student_workspace: workspace || null, local_url: url, weeks, use_saved_key: reuse})});
            const data = await response.json();
            if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Could not save your key. Check the settings.');
            studentConnection = {...data, key: data.key || ''};
        } else {
            await deleteSavedStudentKey();
            studentConnection = provider === 'course' ? {provider, model: '', key: '', url: ''} : {provider, model, key, url, workspace};
        }
        studentRestoreError = '';
        updateStudentModelButton();
        document.getElementById('student-key').value = '';
        document.getElementById('student-model-dialog').close();
    } catch (error) { document.getElementById('student-settings-error').textContent = error.message; }
    finally { apply.disabled = false; }
}
async function studentModelChat(payload) {
    if (studentRestorePromise) await studentRestorePromise;
    if (studentRestoreError) throw Error(studentRestoreError);
    const connection = {...studentConnection};
    payload.student_provider = connection.provider;
    if (connection.saved) payload.use_saved_key = true;
    if (connection.provider !== 'course') payload.student_model = connection.model;
    if (connection.workspace) payload.student_workspace = connection.workspace;
    const headers = {'Content-Type': 'application/json'};
    if (connection.provider !== 'course' && connection.provider !== 'local' && !connection.saved) headers['X-ATLAS-Student-Key'] = connection.key;
    const response = await fetch('/course/' + encodeURIComponent(courseId) + '/chat', {
        method: 'POST', headers, body: JSON.stringify(payload), cache: 'no-store'
    });
    const data = await response.json();
    if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Check the model settings and try again.');
    // No-source answers need no model call, including in local mode.
    if (connection.provider !== 'local' || !data.local_request) return data;
    const localHeaders = {'Content-Type': 'application/json'};
    if (connection.key) localHeaders.Authorization = 'Bearer ' + connection.key;
    let localResponse;
    try {
        localResponse = await fetch(localModelURL(connection.url), {
            method: 'POST', headers: localHeaders, body: JSON.stringify(data.local_request),
            credentials: 'omit', redirect: 'error', signal: AbortSignal.timeout(180000)
        });
    } catch (_) {
        throw Error('Could not reach your local model. Start the local server, allow this ATLAS website in its browser access settings, and allow local-network access if your browser asks. See Model settings for setup links.');
    }
    if (!localResponse.ok) throw Error('Your local server returned error ' + localResponse.status + '. Check the model ID, server token, and available context size. The instructor’s model was not used.');
    const result = await localResponse.json();
    const answer = result.choices?.[0]?.message?.content;
    if (typeof answer !== 'string' || !answer.trim()) throw Error('The local model returned no answer. Check its model and context settings.');
    return {response: answer, sources: data.sources, materials_found: data.materials_found};
}
