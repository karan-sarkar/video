from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

class LabelTextEncode():
    def __init__(self, dataset):
        if dataset == 'kinetics':
            self.change = {
                'clean and jerk': ['weight', 'lift'],
                'dancing gangnam style': ['dance', 'korean'],
                'breading or breadcrumbing': ['bread', 'crumb'],
                'eating doughnuts': ['eat', 'bun'],
                'faceplanting': ['face', 'fall'],
                'hoverboarding': ['skateboard', 'electric'],
                'hurling (sport)': ['hurl', 'sport'],
                'jumpstyle dancing': ['jumping', 'dance'],
                'passing American football (in game)': ['pass', 'american', 'football', 'match'],
                'passing American football (not in game)': ['pass', 'american', 'football', 'park'],
                'petting animal (not cat)': ['pet', 'animal'],
                'punching person (boxing)': ['punch', 'person', 'boxing'],
                's head": 1}': ['head'],
                'shooting goal (soccer)': ['shoot', 'goal', 'soccer'],
                'skiing (not slalom or crosscountry)': ['ski'],
                'throwing axe': ['throwing', 'ax'],
                'tying knot (not on a tie)': ['ty', 'knot'],
                'using remote controller (not gaming)': ['remote', 'control'],
                'backflip (human)': ['backflip', 'human'],
                'blowdrying hair': ['dry', 'hair'],
                'making paper aeroplanes': ['make', 'paper', 'airplane'],
                'mixing colours': ['mix', 'colors'],
                'photobombing': ['take', 'picture'],
                'playing rubiks cube': ['play', 'cube'],
                'pretending to be a statue': ['pretend', 'statue'],
                'throwing ball (not baseball or American football)': ['throw',  'ball'],
                'curling (sport)': ['curling', 'sport'],
            }
        elif dataset == 'activity-net':
            self.change = {
                'Blow-drying_hair': ['dry', 'hair'],
                'Playing_rubik_cube': ['play', 'cube'],
                'Carving_jack-o-lanterns': ['carve', 'pumpkin'],
                'Mooping_floor': ['mop', 'floor'],
                'Ping-pong': ['table', 'tennis'],
                'Plataform_diving': ['diving', 'trampoline'],
                'Polishing_forniture': ['polish', 'furniture'],
                'Powerbocking': ['jump', 'shoes'],
                'Rock-paper-scissors': ['play', 'rock', 'paper', 'scissors'],
            }
        else:
            self.change = {}
    
    def __call__(self, name):
        if name in self.change:
            return self.change[name]
        name = name.lower()
        name_vec_origin = name.split(' ')
        remove = ['a', 'the', 'of', ' ', '', 'and', 'at', 'on', 'in', 'an', 'or',
                  'do', 'using', 'with']         
        name_vec = [n for n in name_vec_origin if n not in remove]
        not_id = [i for i, n in enumerate(name_vec) if n == '(not']
        if len(not_id) > 0:
            name_vec = name_vec[:not_id[0]]
        name_vec = [name.replace('(', '').replace(')', '') for name in name_vec]
        name_vec = self.verbs2basicform(name_vec)
        return tuple(name_vec)
    
    def verbs2basicform(self, words):
        ret = []
        for w in words:
            analysis = wn.synsets(w)
            if any([a.pos() == 'v' for a in analysis]):
                w = WordNetLemmatizer().lemmatize(w, 'v')
            ret.append(w)
        return ret