import { Layout } from './components/Layout';
import { Sidebar } from './components/Sidebar/Sidebar';
import { ChatContainer } from './components/Chat/ChatContainer';
import { FilePreviewModal } from './components/FilePreview/FilePreviewModal';

function App() {
  return (
    <>
      <Layout sidebar={<Sidebar />}>
        <ChatContainer />
      </Layout>
      <FilePreviewModal />
    </>
  );
}

export default App;
