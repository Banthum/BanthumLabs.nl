// Aanmeldformulier beta-test.
// Zolang FORMULIER_ADRES leeg is, opent het formulier een ingevulde mail naar
// info@banthumlabs.nl (geen externe dienst). Met een formulierdienst vul je hier
// het adres in waar het formulier naartoe moet (POST, JSON).
const FORMULIER_ADRES = "";
const MAILADRES = "info@banthumlabs.nl";

const formulier = document.getElementById("beta-formulier");
const melding = document.getElementById("melding");

function toon(tekst, fout) {
  melding.textContent = tekst;
  melding.classList.toggle("fout", !!fout);
}

formulier.addEventListener("submit", async (e) => {
  e.preventDefault();

  const email = formulier.email.value.trim();
  if (!email || !formulier.email.checkValidity()) {
    toon("Vul een geldig e-mailadres in.", true);
    formulier.email.focus();
    return;
  }
  if (!formulier.akkoord.checked) {
    toon("Vink aan dat je akkoord gaat met het gebruik van je gegevens.", true);
    return;
  }

  const gegevens = {
    email,
    batterij: formulier.batterij.value,
    meter: formulier.meter.value,
    contract: formulier.contract.value,
    zon: formulier.zon.value,
    opmerking: formulier.opmerking.value.trim(),
  };

  if (!FORMULIER_ADRES) {
    const tekst =
      "Ik heb interesse in de beta-test van de BMSmachine.\n\n" +
      "E-mail: " + gegevens.email + "\n" +
      "Batterij: " + gegevens.batterij + "\n" +
      "HomeWizard P1-meter: " + gegevens.meter + "\n" +
      "Dynamisch contract: " + gegevens.contract + "\n" +
      "Zonnepanelen: " + gegevens.zon + "\n" +
      (gegevens.opmerking ? "Opmerking: " + gegevens.opmerking + "\n" : "");
    window.location.href =
      "mailto:" + MAILADRES +
      "?subject=" + encodeURIComponent("Interesse beta-test BMSmachine") +
      "&body=" + encodeURIComponent(tekst);
    toon("Je mailprogramma opent met een ingevulde mail. Verstuur die om je interesse door te geven. Opent er niets? Mail dan naar " + MAILADRES + ".");
    return;
  }

  const knop = formulier.querySelector("button");
  knop.disabled = true;
  toon("Bezig met versturen...");
  try {
    const antwoord = await fetch(FORMULIER_ADRES, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(gegevens),
    });
    if (!antwoord.ok) throw new Error(antwoord.status);
    formulier.reset();
    toon("Bedankt! Je interesse is doorgegeven. Je krijgt een mail zodra er meer bekend is over de beta-test.");
  } catch {
    toon("Versturen lukte niet. Mail je interesse naar " + MAILADRES + ".", true);
  } finally {
    knop.disabled = false;
  }
});
