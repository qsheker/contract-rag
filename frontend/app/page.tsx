import { Chat } from "@/components/chat";
import { DocumentUpload } from "@/components/document-upload";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col gap-4 px-4 py-6">
      <header className="space-y-1">
        <h1 className="text-xl font-semibold">contract-rag</h1>
        <p className="text-sm text-muted-foreground">
          Вопросы по юридическим договорам с обязательной ссылкой на пункт и
          страницу.
        </p>
      </header>

      <DocumentUpload />
      <Chat />
    </main>
  );
}
