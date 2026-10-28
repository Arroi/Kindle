import { LinkedSessions } from "../components/LinkedSessions";
import { SectionTitle } from "../components/ui";

export default function Links() {
  return (
    <div>
      <SectionTitle title="Linked sessions" description="Pool another Kindle session with the same GPUs behind this one's URL and API key." />
      <section className="panel max-w-3xl p-5 sm:p-6">
        <LinkedSessions />
      </section>
    </div>
  );
}
