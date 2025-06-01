import { FC, useState, useRef } from 'react';
import MainLayout from '../components/layout/MainLayout';

const AskAIPage: FC = () => {
  const [messages, setMessages] = useState<{ role: 'user' | 'ai'; content: string }[]>([
    { role: 'ai', content: 'Hello! I\'m MediMind AI. How can I assist you with your medical questions today?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showFaqs, setShowFaqs] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Sample FAQs
  const faqs = [
    "What are common symptoms of diabetes?",
    "How can I lower my blood pressure naturally?",
    "What vaccination schedule is recommended for children?",
    "How to identify signs of a stroke?",
    "What causes migraines and how to prevent them?"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const userMessage = { role: 'user' as const, content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setShowFaqs(false);
    setIsLoading(true);

    // Simulate AI response (in a real app, this would be an API call)
    setTimeout(() => {
      // Example response - in production, this would come from your backend
      const aiResponse = {
        role: 'ai' as const,
        content: `Thank you for your question about "${input}". This is a simulated response. In the actual application, I would provide medically accurate information based on your query.`
      };
      setMessages(prev => [...prev, aiResponse]);
      setIsLoading(false);
      scrollToBottom();
    }, 1000);
  };

  const handleFaqClick = (faq: string) => {
    setInput(faq);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      // In a real app, you would handle file upload to your backend
      const fileName = files[0].name;
      setMessages(prev => [
        ...prev,
        { role: 'user', content: `Uploaded file: ${fileName}` },
        { role: 'ai', content: `I've received your file "${fileName}". What specific questions do you have about this medical report?` }
      ]);
      setShowFaqs(false);
      scrollToBottom();
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  return (
    <MainLayout>
      <div className="flex flex-col h-[calc(100vh-200px)] bg-white rounded-xl shadow-md overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-800 p-4">
          <h2 className="text-2xl font-bold text-white">
            Ask<span className="text-green-300">AI</span> - Medical Assistant
          </h2>
          <p className="text-blue-100 text-sm">
            Get answers to your medical questions from our AI-powered medical assistant
          </p>
        </div>

        {/* Message Area */}
        <div className="flex-grow p-4 overflow-y-auto bg-gray-50">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`mb-4 ${message.role === 'user' ? 'flex justify-end' : 'flex justify-start'}`}
            >
              <div
                className={`max-w-[80%] p-3 rounded-lg ${message.role === 'user'
                    ? 'bg-blue-600 text-white rounded-tr-none'
                    : 'bg-white text-gray-800 shadow rounded-tl-none border border-gray-200'
                  }`}
              >
                {message.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start mb-4">
              <div className="bg-white text-gray-500 p-3 rounded-lg shadow rounded-tl-none border border-gray-200">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* FAQs */}
        {showFaqs && (
          <div className="px-4 py-3 bg-gray-100 border-t border-gray-200">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Frequently Asked Questions:</h3>
            <div className="flex flex-wrap gap-2">
              {faqs.map((faq, index) => (
                <button
                  key={index}
                  onClick={() => handleFaqClick(faq)}
                  className="bg-white border border-gray-300 text-gray-700 text-sm px-3 py-1 rounded-full hover:bg-gray-50"
                >
                  {faq}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="p-4 border-t border-gray-200 bg-white">
          <div className="flex items-center">
            <button
              onClick={triggerFileUpload}
              className="p-2 text-gray-500 hover:text-blue-600 focus:outline-none"
              title="Upload medical report"
            >
              📎
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                className="hidden"
                accept=".pdf,.jpg,.jpeg,.png,.doc,.docx"
              />
            </button>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Ask a medical question or upload your report..."
              className="flex-grow p-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={handleSendMessage}
              disabled={!input.trim()}
              className={`p-2 px-4 rounded-r-md ${input.trim()
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
            >
              Send
            </button>
          </div>

          <div className="mt-2 text-xs text-gray-500 flex justify-between items-center">
            <div>
              MediMind AI provides general information only. Consult a healthcare professional for medical advice.
            </div>
            <button
              onClick={() => setShowFaqs(!showFaqs)}
              className="text-blue-600 hover:underline"
            >
              {showFaqs ? 'Hide FAQs' : 'Show FAQs'}
            </button>
          </div>
        </div>
      </div>

      {/* Medical Topics Section */}
      <section className="mt-8 bg-white p-6 rounded-xl shadow-md">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Popular Medical Topics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {['Cardiology', 'Neurology', 'Pediatrics', 'Dermatology', 'Orthopedics', 'Gynecology', 'Oncology', 'Nutrition'].map((topic, index) => (
            <div
              key={index}
              className="bg-blue-50 border border-blue-100 p-4 rounded-lg hover:bg-blue-100 cursor-pointer transition-colors"
            >
              <h3 className="font-medium text-blue-800">{topic}</h3>
              <p className="text-xs text-gray-600 mt-1">Common questions and information</p>
            </div>
          ))}
        </div>
      </section>

      {/* Disclaimer */}
      <section className="mt-6 p-4 border border-orange-200 bg-orange-50 rounded-lg">
        <div className="flex items-start">
          <div className="text-orange-500 mr-3">⚠️</div>
          <div>
            <h3 className="font-bold text-orange-800">Important Disclaimer</h3>
            <p className="text-sm text-orange-700">
              MediMind AI is designed to provide general information only and should not be used as a substitute for professional medical advice,
              diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may
              have regarding a medical condition.
            </p>
          </div>
        </div>
      </section>
    </MainLayout>
  );
};

export default AskAIPage;