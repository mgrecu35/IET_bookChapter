%%
%% This is file `book.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% classes.dtx  (with options: `book')
%% 
%% This is a generated file.
%% 
%% Copyright 1993 1994 1995 1996 1997 1998 1999 2000
%% The LaTeX3 Project and any individual authors listed elsewhere
%% in this file.
%% 
%% This file was generated from file(s) of the LaTeX base system.
%% --------------------------------------------------------------
%% 
%% It may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.2
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.2 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%% 
%% This file may only be distributed together with a copy of the LaTeX
%% base system. You may however distribute the LaTeX base system without
%% such generated files.
%% 
%% The list of all files belonging to the LaTeX base distribution is
%% given in the file `manifest.txt'. See also `legal.txt' for additional
%% information.
%% 
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{book}
              [2000/05/19 v1.4b
 Standard LaTeX document class]
\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagetrue
\newif\if@openright
\newif\if@mainmatter \@mainmattertrue
\if@compatibility\else
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility
\@openrighttrue
\else
\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}
\ExecuteOptions{letterpaper,10pt,twoside,onecolumn,final,openright}
\ProcessOptions
\input{bk1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}


\setlength\parskip{0pt}
\setlength\parindent{18pt}

\newdimen\@bls
\@bls=\baselineskip
\bigskipamount=\@bls \@plus 0.3\@bls \@minus 0.3\@bls
\medskipamount=0.5\bigskipamount
\smallskipamount=0.25\bigskipamount


\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\quad\enskip \slshape\leftmark\hfill}%
      \def\@oddhead{\hfill{\slshape\rightmark}\quad\enskip \thepage}%
      \let\@mkboth\markboth}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{\hfill{\slshape\rightmark}\quad\enskip\thepage}%
    \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markright {%\MakeUppercase
                 {%
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}}}}
\fi
\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\thepage\hfil\slshape\leftmark}%
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\@gobbletwo
    \let\chaptermark\@gobble
    \let\sectionmark\@gobble
    }
  \if@titlepage
  \newcommand\maketitle{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
%  \vskip 60\p@
\vspace*{23pt}
%  \begin{center}%
\begin{flushleft}
    {\fontsize{30}{32}\selectfont\sf %\LARGE 
    \@title 
    \par}%
    \vskip 2.35em%
    {%\large
     %\lineskip .75em%
      %\begin{tabular}[t]{c}%
      \fontsize{14}{16}\selectfont\sf \@author
      %\end{tabular}
      \vfill
      {\fontsize{10}{10}\selectfont\sf The Institution of Engineering and Technology}\vspace*{-11pt}  
      \par}%
      %\vskip 1.5em%
    %{\large \@date \par}%       % Set date in \large size.
\end{flushleft}
%  \end{center}
\par
  \@thanks
  \vfil\null
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\else
\newcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{empty}\@thanks%plain
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}

\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}
\fi
\newcommand*\chaptermark[1]{}
\setcounter{secnumdepth}{2}
\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thechapter {\@arabic\c@chapter}
\renewcommand \thesection {\thechapter.\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection .\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\newcommand\@chapapp{\chaptername}
\newcommand\frontmatter{%
    \cleardoublepage
  \@mainmatterfalse
  \pagenumbering{roman}}
\newcommand\mainmatter{%
    \cleardoublepage
  \@mainmattertrue
  \pagenumbering{arabic}}
\newcommand\backmatter{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \@mainmatterfalse}
\newcommand\part{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \thispagestyle{empty}%{plain}%
  \if@twocolumn
    \onecolumn
    \@tempswatrue
  \else
    \@tempswafalse
  \fi
  \null\vfil
  \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {%\centering
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >-2\relax
       %\huge
        \vspace*{6.1pc}
       \noindent{\fontsize{14}{14}\selectfont\itshape \partname~\thepart}
       \par
       \vskip 12\p@
     \fi
     %\Huge 
     \noindent{\fontsize{16}{18}\selectfont\bfseries #2}\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \Huge \bfseries #1\par}%
    \@endpart}
\def\@endpart{\vfill\newpage
              \if@twoside
               \if@openright
                \null
                \thispagestyle{empty}%
                \newpage
               \fi
              \fi
              \if@tempswa
                \twocolumn
              \fi}
\newcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{empty}%{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\def\cauthor#1{\let\thanks\footnote\def\@cauthor{#1\vspace*{12pt}}}
\def\@makechapterhead#1{\let\thanks\footnote%
\def\@makefnmark{{\@textsuperscript{\normalfont{\itshape\@thefnmark}}}}%
\renewcommand\@makefntext[1]{\def\@makefnmark{{\@textsuperscript{\normalfont{\@thefnmark}}}}%
    \parindent\z@% 1em%
    \noindent
    {\rm\@makefnmark}##1}
\iffalse  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \huge\bfseries \@chapapp\space \thechapter
        \par\nobreak
        \vskip 20\p@
      \fi
    \fi
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 40\p@
  }\fi\chaptermark{#1}
\vspace*{-18.5pt}
\noindent\rule{\textwidth}{1pt}\vspace*{34pt}
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        {\centering\fontsize{14}{14}\selectfont\it 
         \@chapapp\space \thechapter
        \par\nobreak}\vspace*{6pt}
      \fi
    \fi
    \interlinepenalty\@M
    {\centering\fontsize{16}{16}\selectfont\bf
    #1\par\nobreak}\vspace*{8pt}%
    {\centering\fontsize{14}{16}\selectfont\itshape {\@cauthor}\par}
}\vskip.01pt  
\noindent\rule{\textwidth}{1pt}
\vspace*{12pt}
\global\let\@cauthor\@empty
\global\let\@thanks\@empty}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
% \vspace*{50\p@}%
%  {\parindent \z@ \raggedright
%    \normalfont
%    \interlinepenalty\@M
%    \Huge \bfseries  #1\par\nobreak
%    \vskip 40\p@
%  }
\vspace*{-21pt}
\noindent\rule{\textwidth}{1pt}\vspace*{31pt}
  {\parindent \z@ \raggedright \normalfont
    \interlinepenalty\@M
    {\centering\fontsize{16}{16}\selectfont\bf
    #1\par\nobreak}\vspace*{12.5pt}%
  }\vskip.01pt
\noindent\rule{\textwidth}{1pt}
\vspace*{12pt}\vskip.01pt
}
\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\Large\bfseries}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\large\bfseries}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}
\if@twocolumn
  \setlength\leftmargini  {1.5em}
\else
  \setlength\leftmargini  {1.5em}
\fi
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\setlength\leftmargini{1.5em}\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel\setlength\labelsep{.5em}}\itemsep0pt\parskip0pt}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\if@compatibility
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
\newcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset {equation}{chapter}
\renewcommand\theequation
  {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@equation}
\newcounter{figure}[chapter]
\renewcommand \thefigure
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1[#2]#3{\vskip4.5pt\par\fontsize{10}{12}\selectfont\raggedright
        \addcontentsline{\csname ext@#1\endcsname}
        {#1}{\protect\numberline{\csname fnum@#1\endcsname}
        {\ignorespaces #2}}
        \begingroup
        \@parboxrestore
%        \footnotesize
        \@makecaption{\csname fnum@#1\endcsname}{#3}\par\vskip12.5pt
        \endgroup}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{10004.5}
\setcounter{tocdepth}{2}
\newcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{%
           {\@rhbooktitle}}{\contentsname}}\setcounter{page}{5}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{1.6em \@plus\p@}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \bfseries\partname~#1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
    \vspace{-6\p@}%
  \fi}
\newcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \vspace{2\p@}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\newcommand*\l@schapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
%    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\newcommand*\l@bschapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.5em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\newcommand*\l@section{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*\l@subsection{\@dottedtocline{2}{3.8em}{3.2em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{10em}{5em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{12em}{6em}}
\newcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\@rhbooktitle}%
              {\listfigurename}}%
    \addvspace{-10pt}%
    \addcontentsline{toc}{schapter}{\listfigurename}
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{%
          \@rhbooktitle}{\listtablename}}%
    \addvspace{-10pt}%
    \addcontentsline{toc}{schapter}{\listtablename}
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\section*{\bibname%
        \@mkboth{\@rhbooktitle}{\MakeUppercase\bibname}}
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \itemsep0pt\parskip0pt
      \sloppy
      \clubpenalty10000
      \@clubpenalty \clubpenalty
      \widowpenalty10000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\@makeschapterhead{\indexname}]%
                \@mkboth{\@rhbooktitle}%
                        {\indexname}%
                \thispagestyle{empty}%{plain}
                \parindent\z@
                \parskip\z@ \@plus \z@\relax
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 30\p@}
\newcommand\subitem{\@idxitem \hspace*{10\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{20\p@}}
\newcommand\indexspace{\par \vskip 12\p@ \@plus3\p@ \@minus2\p@\relax}
\renewcommand\footnoterule{%
%  \kern-3\p@
%  \hrule\@width.4\columnwidth
%  \kern2.6\p@
}
\@addtoreset{footnote}{chapter}
\newcommand\@makefntext[1]{%
    \parindent\z@% 1em%
    \noindent
    {\@makefnmark}#1}
\newcommand\contentsname{Contents}
\newcommand\listfigurename{List of figures}
\newcommand\listtablename{List of tables}
\newcommand\bibname{References}%{Bibliography}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\chaptername{Chapter}
\newcommand\appendixname{Appendix}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{headings}
\pagenumbering{arabic}
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi
%%%%%%
\setlength{\paperheight}{234mm}
\setlength{\paperwidth}{156mm}
\newif\if@cropmarkson \@cropmarksontrue
\newif\if@restonecol
\RequirePackage{times,crop,epsfig,amsmath,amsfonts,amsthm,amsbsy,latexsym,amssymb,fleqn,framed,color}
\usepackage[figuresright]{rotating}
\usepackage{mathptmx}
\usepackage{enumerate}
\usepackage{url}
\usepackage{index}
\DeclareOption{centre}{\PassOptionsToPackage{a4center}{crop}}
\DeclareOption{crop}{\PassOptionsToPackage{cam}{crop}\global\@cropmarksontrue}
\DeclareOption{nocrop}{\PassOptionsToPackage{off}{crop}\global\@cropmarksonfalse}
\ExecuteOptions{crop,centre}
\ProcessOptions
\definecolor{shadecolor}{gray}{.9} % 10%
\renewcommand*\CROP@@ulc{%
%  \begin{picture}(0,0)\unitlength\p@\thinlines
%  \put(-42,0){\line(1,0){30}}
%  \put(0,42){\line(0,-1){30}}
%  \end{picture}
}
\renewcommand*\CROP@@urc{%
%  \begin{picture}(0,0)\unitlength\p@\thinlines
%  \put(42,0){\line(-1,0){30}}
%  \put(0,42){\line(0,-1){30}}
%  \end{picture}
}
\renewcommand*\CROP@@llc{%
%  \begin{picture}(0,0)\unitlength\p@\thinlines
%  \put(-42,0){\line(1,0){30}}
%  \put(0,-42){\line(0,1){30}}
%  \end{picture}
}
\renewcommand*\CROP@@lrc{%
%  \begin{picture}(0,0)\unitlength\p@\thinlines
%  \put(42,0){\line(-1,0){30}}
%  \put(0,-42){\line(0,1){30}}
%  \end{picture}
}
\cropdef\CROP@@ulc\CROP@@urc\CROP@@llc\CROP@@lrc{cam}
\if@cropmarkson \crop[cam,info] \fi

\newcount\minute
\newcount\hour
\def\currenttime{%
\minute\time
\hour\minute
\divide\hour60
\the\hour: \multiply\hour60\advance\minute-\hour\the\minute}

\setlength\topmargin{3.32pc}
\addtolength\topmargin{-1in}
\setlength\headheight{8\p@}
\setlength\topskip{8\p@}
\setlength\headsep{2pc}
\addtolength\headsep{-\topskip}
\setlength\footskip{24\p@}
\setlength\maxdepth{.5\topskip}
\setlength\textwidth{118mm}
\setlength\textheight{44.75pc}

\addtolength\textheight{\topskip}
\setlength\oddsidemargin{22mm}
\addtolength\oddsidemargin{-1in}
\setlength\@tempdima{\paperwidth}
\addtolength\@tempdima{-\textwidth}
\addtolength\@tempdima{-22mm}
\setlength\evensidemargin{\@tempdima}
\addtolength\evensidemargin{-1in}
\setcounter{secnumdepth}{3}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0pt}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\clubpenalty 10000
\widowpenalty 10000

\renewcommand\section{%
\@startsection {section}{1}{\z@}%
{-18\p@ \@plus -1pt \@minus -2pt}%
{12\p@}%
{\fontsize{12}{13}\selectfont\bfseries\raggedright}}
\renewcommand\subsection{%
\@startsection{subsection}{2}{\z@}%
{-18pt \@plus -1pt \@minus -2pt}%
{4pt}%
{\fontsize{12}{13}\selectfont\itshape\raggedright}}
\renewcommand\subsubsection{%
\@startsection{subsubsection}{3}{\z@}%
{-12pt \@plus -1pt \@minus -2pt}%
{2pt}%
{\fontsize{11}{12}\selectfont\bfseries\raggedright}}
\renewcommand\paragraph{%
\@startsection{paragraph}{4}{\z@}%
{-12pt \@plus -1pt \@minus -2pt}%
{.00001pt}%
{\fontsize{11}{12}\selectfont\itshape\raggedright}}

\newcommand{\boxhead}[1]{{\fontsize{11}{13}\selectfont\bfseries\raggedright #1 \par}\vskip2pt}
\newenvironment{boxes}[2]{\begin{shaded}{#1}{\noindent #2}\end{shaded}}

\def\ps@newheadings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\fontsize{10}{10}\selectfont{\thepage}\quad {\itshape\leftmark}\hfil}%
      \def\@oddhead{\fontsize{10}{10}\selectfont\hfil{\itshape\rightmark}\quad {\thepage}}%
      \let\@mkboth\markboth}
%\ps@newheadings
\fboxsep=12\p@


%With crop mark %use dvips -t "a4" filename.dvi for create dvi to ps/pdf
%\hoffset1in\voffset-1.5in

%Without the Crop mark %pdflatex filename.tex (or) dvips filename.dvi
%\hoffset0in\voffset0in

\renewcommand\mathindent{24pt}

%%% 18th Nov 2016
%\reversemarginpar\marginparwidth30pt
%\def\styletag#1#2{\marginpar{\vspace*{#2pt}\fboxsep2pt\fbox{\hbox{\vbox{\hsize2pc{\bf #1}}}}}}
\def\styletag#1#2{}%{\marginpar{\vspace*{#2pt}\fboxsep2pt\fbox{\hbox{\vbox{\hsize2pc{\bf #1}}}}}}
%%%

\def\contriauthor#1{\noindent{\bf #1}}

\def\toprule{\noalign{\ifnum0=`}\fi\hrule \@height 1pt \hrule \@height 5pt \@width 0pt \futurelet
   \@tempa\@xhline}
\def\botrule{\noalign{\ifnum0=`}\fi \hrule \@height 3pt \@width 0pt \hrule \@height 1pt \futurelet
   \@tempa\@xhline}
\def\midrule{\noalign{\ifnum0=`}\fi \hrule \@height 3pt \@width 0pt \hrule \@height .5pt \hrule \@height 5pt \@width 0pt \futurelet
   \@tempa\@xhline}

\def\TCH#1{{\bfseries #1}}

\newif\if@rotate \@rotatefalse
\newif\if@rotatecenter \@rotatecenterfalse
\def\rotatecenter{\global\@rotatecentertrue}
\def\rotateendcenter{\global\@rotatecenterfalse}
\def\rotate{\global\@rotatetrue}
\def\endrotate{\global\@rotatefalse}
\newdimen\rotdimen
\def\rotstart#1{\special{ps: gsave currentpoint currentpoint translate
    #1 neg exch neg exch translate}}
\def\rotfinish{\special{ps: currentpoint grestore moveto}}
\def\rotl#1{\rotdimen=\ht#1\advance\rotdimen by \dp#1
    \hbox to \rotdimen{\vbox to\wd#1{\vskip \wd#1
    \rotstart{270 rotate}\box #1\vss}\hss}\rotfinish}
\def\rotr#1{\rotdimen=\ht #1\advance\rotdimen by \dp#1
    \hbox to \rotdimen{\vbox to \wd#1{\vskip \wd#1
    \rotstart{90 rotate}\box #1\vss}\hss}\rotfinish}

\long\def\@makecaption#1#2{\vskip4.5pt\fontsize{10}{12}\selectfont\raggedright\itshape
\sbox\@tempboxa{{{#1}\hskip9pt}\ignorespaces{#2}}
\ifdim\wd\@tempboxa > \hsize\par \@hangfrom{{#1}\hskip9pt}\ignorespaces{#2} \else
\hb@xt@\hsize{\hfil{\box\@tempboxa}\hfil}
\fi\vspace*{2pt}}

\newbox\all \newbox\lastpart
\long\def\@maketablecaption#1#2{\fontsize{10}{12}\selectfont\raggedright
        \setbox\@tempboxa\vbox{\itshape\@hangfrom{{#1}\hskip9pt}\ignorespaces{#2}}
\vskip2pt
        \ifdim \wd\@tempboxa >\hsize
                \par\unhbox\@tempboxa %
                \else
               \hbox to\hsize{\box\@tempboxa\hfil}
        \fi\vskip14.5pt}

\def\table{\let\@makecaption\@maketablecaption\@float{table}}

\newdimen\tempdime
\newbox\temptbox
\newenvironment{tabnote}[1]{\vspace*{4pt}\par\noindent\fontsize{8}{9}\selectfont #1}{\par}
\newenvironment{processtable}[3]{{\setbox\temptbox=\hbox{{#2}}%
\tempdime\wd\temptbox\@processtable{#1}{#2}{#3}{\tempdime}}}
{\relax}

\newenvironment{@processtable}[4]{%
\if@rotate
\setbox4=\vbox to \textwidth{\vss\hbox to \textheight{\hss%
\begin{minipage}{#4}%
\caption{#1}{\fontsize{9}{10}\selectfont\baselineskip=11pt #2}%
\tabnote{\fontsize{8}{9}\selectfont #3\par}%
\end{minipage}}\vss}
\else
\setbox4=\hbox to \hsize{\vbox{\begin{center}
\begin{minipage}[t]{#4}%
\caption{#1}{\fontsize{9}{10}\selectfont\baselineskip=11pt {#2}}
\tabnote{\fontsize{8}{9}\selectfont #3\ \par}
\end{minipage}
\end{center}
}}
\fi
\if@rotate\rotl{4}\else\box4\fi}
{\relax}

\newenvironment{unnumlist}
               {\setlength\leftmargini{1.5em}\list{}{\labelwidth\z@\itemindent-\leftmargin}\itemsep0pt\parskip0pt}
               {\endlist}

%%% LISTS %%
\newdimen\labelwidthi
\newdimen\labelwidthii
\newdimen\labelwidthiii
\newdimen\labelwidthiv
\def\normal@labelsep{1.25em}
\labelsep\normal@labelsep
\leftmargini\labelwidthi    \advance\leftmargini\labelsep
\leftmarginii\labelwidthii  \advance\leftmarginii\labelsep
\leftmarginiii\labelwidthiii \advance\leftmarginiii\labelsep
\leftmarginiv\labelwidthiv  \advance\leftmarginiv\labelsep
\def\setleftmargin#1#2{\settowidth{\@tempdima}{#2}\labelsep\normal@labelsep
  \csname labelwidth#1\endcsname\@tempdima
  \@tempdimb\@tempdima \advance\@tempdimb\labelsep
  \csname leftmargin#1\endcsname\@tempdimb}
\def\@listI{\leftmargin\leftmargini
  \labelwidth\labelwidthi \labelsep\normal@labelsep\addtolength\leftmargin{6pt}
  \topsep06pt \@plus1\p@ \@minus1\p@
  \partopsep\z@ \parsep\z@ \itemsep\z@
  \listparindent 1em}
\def\@listii{\leftmargin\leftmarginii
  \labelwidth\labelwidthii \labelsep\normal@labelsep
  \topsep\z@ \partopsep\z@ \parsep\z@ \itemsep\z@
  \listparindent 1em}
\def\@listiii{\leftmargin\leftmarginiii
  \labelwidth\labelwidthiii \labelsep\normal@labelsep
  \topsep\z@ \partopsep\z@ \parsep\z@ \itemsep\z@
  \listparindent 1em}
\def\@listiv{\leftmargin\leftmarginiv
  \labelwidth\labelwidthiv \labelsep\normal@labelsep
  \topsep\z@ \partopsep\z@ \parsep\z@ \itemsep\z@
  \listparindent 1em}
\let\@listi\@listI
\@listi
\def\left@label#1{{#1}\hss}
\def\right@label#1{\hss\llap{#1}}
\def\thick@label#1{\hspace\labelsep #1}
\newcount\@maxlistdepth
\@maxlistdepth=4
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\def\labelitemii{--\hss}%{\fontsize{8}{8}\selectfont${\symdia}$}
\def\labelenumi{\theenumi}        \def\theenumi{\arabic{enumi}.}
\def\labelenumii{(\alph{enumii})}   \def\theenumii{\theenumi\alph{enumii}}
\def\labelenumiii{(\roman{enumiii})}\def\theenumiii{\theenumii.\roman{enumiii}}
\def\labelenumiv{(\Alph{enumiv})}   \def\theenumiv{\theenumiii.\Alph{enumiv}}



\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{remark}{Remark}
\newtheorem{example}{Example}
\newtheorem{property}{Property}
\newtheorem{definition}{Definition}

\def\UrlFont{\rmfamily}

\def\ietseries#1{\vspace*{-12pt}\noindent{\sf\bfseries{#1}}}

\def\halftitle#1{\vspace*{22pt}%
\begin{flushleft}%
{\fontsize{30}{32}\selectfont\sf #1 \par}%
\end{flushleft}}

\newcommand\imprintpage[1]{{\par\fontsize{8}{10}\selectfont\sf #1}}

\def\rhbooktitle#1{\def\@rhbooktitle{#1}}

\makeindex
\endinput
%%
%% End of file `book.cls'.
