(* Build doped_gamma.nb from src/: wolframscript -file build.wl nb=doped_gamma.nb src/*.wl  (options: load=x.mx, dump=y.mx) *)
args = Rest[$ScriptCommandLine];
opt[name_] := Replace[SelectFirst[args, StringStartsQ[#, name <> "="] &, None], s_String :> StringDrop[s, StringLength[name] + 1]];
If[opt["load"] =!= None, Get[opt["load"]]];
args = Rest[$ScriptCommandLine]; $checks = If[ListQ[$checks], $checks, {}];
files = Select[args, ! StringContainsQ[#, "="] &];
blocks[file_] := Module[{lines = ReadList[file, String, NullRecords -> True], out = {}, style = None, buf = {}},
  Do[If[StringMatchQ[l, "(* ::" ~~ __ ~~ ":: *)"],
    If[style =!= None, AppendTo[out, {style, buf}]]; style = StringTake[l, {6, -6}]; buf = {},
    AppendTo[buf, l]], {l, lines}];
  If[style =!= None, AppendTo[out, {style, buf}]]; out];
cells = {};
Do[Do[Module[{style = b[[1]], lines = DeleteCases[b[[2]], ""]},
    If[style === "Input",
     Module[{code = StringRiffle[lines, "\n"], exprs, outs = {}},
      AppendTo[cells, Cell[code, "Input"]];
      exprs = ReadList[StringToStream[code], Hold[Expression]];
      Do[Module[{t, r},
        {t, r} = AbsoluteTiming[TimeConstrained[ReleaseHold[e], 10, $TimedOut]];
        If[t > 2, Print["SLOW ", Round[t, 0.1], "s: ", StringTake[ToString[e, InputForm], UpTo[150]]]];
        If[r === $TimedOut, Print["TIMEOUT: ", StringTake[ToString[e, InputForm], UpTo[150]]]];
        If[r =!= Null, AppendTo[cells, Cell[BoxData[ToBoxes[r, StandardForm]], "Output"]];
         Print[StringTake[ToString[r, InputForm], UpTo[300]]]]], {e, exprs}]],
     AppendTo[cells, Cell[StringRiffle[StringReplace[lines, StartOfString ~~ "(*" ~~ s___ ~~ "*)" ~~ EndOfString :> s], "\n"], style]]]],
   {b, blocks[f]}], {f, files}];
If[opt["dump"] =!= None, DumpSave[opt["dump"], "Global`"]];
If[opt["nb"] =!= None,
  cells = UsingFrontEnd[cells /. Cell[code_String, "Input"] :> Cell[First[FrontEndExecute[FrontEnd`UndocumentedTestFEParserPacket[code, False]]], "Input"]];
  Put[Notebook[cells, StyleDefinitions -> "Default.nb"], opt["nb"]]];
