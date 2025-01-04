from environment import CheckersEnv
import matplotlib.pyplot as plt

def main():
    env = CheckersEnv()
    # plt.imshow(env.render('rgb_array'))
    env.render('human')

if __name__ == '__main__':
    main()

