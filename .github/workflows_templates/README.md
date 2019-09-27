# Workflows Templates

The Workflows we use have quite a lot of code repetition. The way we get around this is to use a templating system.

The templating system is a custom (pretty simple and probably pretty inefficient) one. No good reason for this, maybe one day we'll switch to a fancier one.

It operates by taking each `.template` file and substituting strings of the form `<<example>>` until it runs out of substitutions to make. Substitutions can be defined inside substitutions arbitrarily complicatedly. No attempt is made to detect infinite descents.

Run with `python3 from_templates.py`. The results will automatically be placed in the sibling directory `workflows`, overwriting anything already present.

Common values for template parameters are specified in `from_templates.py`. Values specific to each template are defined with the `.template` file in the `Arguments` section at the top of the file. In particular nearly every template will want to define values for:
* `event_name`: The event that triggers this Workflow. Currently it is only possible to trigger a Workflow via either:
  * Just `repository_dispatch` or
  * `repository_dispatch` and precisely one other event; this other event is `<<event_name>>`. Note that often this value will need to be set even if only triggering via `repository_dispatch`; just set it to some dummy value e.g. `-no-event-`.
* `event_cond`: Any further condition that must be met when running the Workflow because of `<<event_name>>`. e.g. set to `true` for no further conditions, or `false` to only trigger the event via `repository_dispatch`.
* `trigger`: In principle every Workflow can by default be triggered manually via a `repository_dispatch` event. (See `workflows/trigger.sh` for actually sending these events.) This argument specifies the string associated with the the trigger, and is typically related to the file name.

### How much code reuse is there anyway?
(i.e. why is this all necessary?)

Answer: A reasonable amount, mostly on two fronts.

Firstly, in terms of the command line code that is run. Actually compiling/testing etc. on each platform is a somewhat finicky process, so it's important to standardise this. For example:
* Windows requires chaining commands with `&&`.
* Windows requires running a `vcvars` file prior to any compilation.
* Mac requires the whole script to be run with `sudo`.
* etc.

Secondly, in terms of running Workflows manually via `repository_dispatch`. Also a somewhat finicky process. This is what is responsible for the the massive `if` statements you'll see in the generated code. More than that, if this process is to work then it must be applied consistently; another benefit of templating.