import { getToken } from '@vercel/connect';
import { generateText } from 'ai';

const token = await getToken('jev/acme-jev');
if (!token) {
  throw new Error('Jev connector jev/acme-jev returned an empty token');
}

const { text } = await generateText({
  model: 'openai/gpt-5.5',
  prompt: 'Invent a new holiday and describe its traditions.',
});

console.log(text);
