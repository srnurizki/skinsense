# Import Libraries
import requests
import streamlit as st
import base64

# Config
FASTAPI_BASE_URL = st.secrets.get('FASTAPI_BASE_URL', 'http://127.0.0.1:8000')
SKIN_TYPE_OPTIONS = {
    'combination': 'Kombinasi',
    'dry': 'Kering',
    'normal': 'Normal',
    'oily': 'Berminyak',
}
SKIN_CONCERN_OPTIONS = {
    'dark spots': 'Noda/Bintik Hitam',
    'inflammatory acne': 'Jerawat Meradang',
    'non inflammatory acne black heads': 'Komedo Hitam/Blackheads',
    'non inflammatory acne white heads': 'Komedo Putih/Whiteheads',
    'pigmentation': 'Pigmentasi Kulit',
    'pores': 'Pori-Pori',
    'redness': 'Kemerahan',
    'wrinkles': 'Kerutan',
}

st.set_page_config(page_title='Aphrodia AI', page_icon='🫰', layout='wide')
st.title('Say Hi to Aphrodia 🫰')
st.markdown("**AI Personalisasi Skincare** untuk Menjawab Kebutuhan Kulitmu!")
st.markdown("""
    <style>
    [data-testid="stChatMessageContent"] img {
        max-width: 150px !important;
        max-height: 150px !important;
        object-fit: contain !important; 
        border-radius: 8px; 
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

    html, body, .stApp, p, h1, h2, h3, h4, h5, h6, label, div[data-testid="stMarkdownContainer"] > p {
        font-family: 'DM Sans', sans-serif !important;
        letter-spacing: 0.3px !important;
    }
    
    .material-symbols-rounded, 
    [data-testid="stIconMaterial"], 
    i.material-icons, 
    svg {
        font-family: "Material Symbols Rounded" !important;
        letter-spacing: normal !important;
    }
    background-color: #f4f6f9 !important;
    </style>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'skin_type' not in st.session_state:
    st.session_state.skin_type = None
if 'skin_concern' not in st.session_state:
    st.session_state.skin_concern = None
if 'skin_concern_conf' not in st.session_state:
    st.session_state.skin_concern_conf = None
if 'skin_type_conf' not in st.session_state:
    st.session_state.skin_type_conf = None
if 'image_url' not in st.session_state:
    st.session_state.image_url = None
if 'zone_details' not in st.session_state:
    st.session_state.zone_details = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'pending_prompt' not in st.session_state:
    st.session_state.pending_prompt = None

# SkinSense AI
with st.popover('🤖 Aphrodia SkinSense AI'):
    st.info('Lihat lurus ke kamera. Pastikan wajah tegak, tidak tertutup, dan cahaya di sekitar cukup ☺️')
    img_file = st.camera_input(' ', label_visibility='collapsed')

    if img_file:
        with st.spinner('🪄 Aphrodia sedang menganalisis...'):
            response = requests.post(
                f'{FASTAPI_BASE_URL}/predict/',
                data=img_file.getvalue(),
                headers={'content-type': 'image/jpeg'})

        if response.status_code == 422:
            st.error(response.json().get('detail', 'Wajah kamu gak terdeteksi 😔'))
        elif response.status_code == 200:
            result = response.json()
            st.session_state.result = result
            st.session_state.skin_type = result.get('skin_type')
            st.session_state.skin_concern = result.get('skin_concern')
            st.session_state.skin_type_conf = result.get('skin_type_confidence', 0.0)
            st.session_state.skin_concern_conf = result.get('skin_concern_confidence', 0.0)
            st.session_state.image_url = result.get('image_url')
            st.session_state.zone_details = result.get('zone_details')

    if st.session_state.skin_type:
        st.subheader('Hasil Analisis')
        if st.session_state.zone_details:
            st.markdown('**Detail Analisis Wajah:**')
            cols = st.columns(6)

            with cols[0]:
                st.image(st.session_state.image_url, use_container_width=True)
                st.caption('**Masalah Kulit Terindikasi (Conf. %)**')

                if st.session_state.skin_concern_conf:
                    st.caption(f'{st.session_state.skin_concern} ({st.session_state.skin_concern_conf * 100:.1f}%)')
                else:
                    st.caption(f'{st.session_state.skin_concern}')

            for i, (zone_name, details) in enumerate(st.session_state.zone_details.items()):
                with cols[i + 1]:
                    image_bytes = base64.b64decode(details['image_b64'])
                    st.image(image_bytes, use_container_width=True)

                    st.caption(f'**{zone_name}**')
                    st.caption(f"{details['prediction']} ({details['confidence'] * 100:.2f}%")

        st.markdown('---')

        col1, col2 = st.columns(2)

        st_keys = list(SKIN_TYPE_OPTIONS.keys())
        st_labels = list(SKIN_TYPE_OPTIONS.values())
        sc_keys = list(SKIN_CONCERN_OPTIONS.keys())
        sc_labels = list(SKIN_CONCERN_OPTIONS.values())

        with col1:
            selected_st_label = st.selectbox(
                'Jenis Kulit', st_labels,
                index=st_keys.index(st.session_state.skin_type))
            st.session_state.skin_type = st_keys[st_labels.index(selected_st_label)]

        with col2:
            selected_sc_label = st.selectbox(
                'Masalah Kulit', sc_labels,
                index=sc_keys.index(st.session_state.skin_concern))
            st.session_state.skin_concern = sc_keys[sc_labels.index(selected_sc_label)]

        col_ask, col_feedback = st.columns(2)

        with col_ask:
            if st.button('Tanya Aphrodia'):
                message = (f'Halo, Aphrodia!. Aku punya tipe kulit {SKIN_TYPE_OPTIONS[st.session_state.skin_type]} '
                           f'dengan masalah {SKIN_CONCERN_OPTIONS[st.session_state.skin_concern]}. '
                           f'Minta saran produk buat aku dong!')
                st.session_state.pending_prompt = message
                st.rerun()

        with col_feedback:
            if st.button('Simpan Edit'):
                original_result = st.session_state.result or {}

                requests.post(f'{FASTAPI_BASE_URL}/feedback/', json={
                    'image_url': st.session_state.image_url,
                    'predicted_skin_type': original_result.get('skin_type', ''),
                    'corrected_skin_type': st.session_state.skin_type,
                    'predicted_skin_concern': original_result.get('skin_concern', ''),
                    'corrected_skin_concern': st.session_state.skin_concern,
                })
                st.success('Tersimpan!')

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

user_input = st.chat_input('Tanyakan soal kulit kamu...')

prompt = None
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif user_input:
    prompt = user_input

if prompt:
    chat_history = st.session_state.messages.copy()

    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user', avatar='🙂'):
        st.markdown(prompt)

    with st.chat_message('assistant', avatar='❤️'):
        with st.spinner('Bentar ya, Aphrodia lagi menalar...'):
            try:
                response = requests.post(
                    f'{FASTAPI_BASE_URL}/chat/',
                    json={'message': prompt, 'history': chat_history},
                    stream=True,
                    timeout=300)

                if response.status_code != 200:
                    st.error(f"Error {response.status_code}: {response.text}")
                    st.session_state.messages.append({'role': 'assistant', 'content': f"Error: {response.status_code}"})
                else:
                    def stream_response():
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                text = chunk.decode('utf-8', errors='ignore')
                                if '[DONE]' in text:
                                    text = text.replace('[DONE]', '')
                                    if text:
                                        yield text
                                    break
                                yield text

                    full_response = st.write_stream(stream_response())

                    if not full_response:
                        full_response = "Maaf, Aphrodia udah proses permintaan kamu, tapi ada kegagalan di sistem 🙏."
                        st.write(full_response)

                    st.session_state.messages.append({'role': 'assistant', 'content': full_response})

            except Exception as e:
                error_msg = f"Koneksi backend gagal: {e}"
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})