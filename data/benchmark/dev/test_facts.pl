%% Tiny test dataset for verifying effective_distance.pl
%% Structure:
%%   Science (root)
%%     └── Physics
%%           ├── Quantum mechanics
%%           │     └── "Higgs boson" (article)
%%           │     └── "Bose-Einstein statistics" (article)
%%           └── Classical mechanics
%%                 └── "Newton's laws of motion" (article)
%%           └── "Special relativity" (article, directly in Physics)
%%
%% Also: Quantum mechanics → Theoretical physics → Physics (alternative path)

% Category hierarchy
category_parent('Quantum mechanics', 'Physics').
category_parent('Quantum mechanics', 'Theoretical physics').
category_parent('Classical mechanics', 'Physics').
category_parent('Theoretical physics', 'Physics').
category_parent('Physics', 'Science').
category_parent('Physics', 'Natural sciences').
category_parent('Natural sciences', 'Science').

% Additional category hierarchy
category_parent('Statistical mechanics', 'Physics').

% Article → category
article_category('Higgs boson', 'Quantum mechanics').
article_category('Bose-Einstein statistics', 'Quantum mechanics').
article_category('Bose-Einstein statistics', 'Statistical mechanics').
article_category('Newton\'s laws of motion', 'Classical mechanics').
article_category('Special relativity', 'Physics').

% Root
root_category('Science').
