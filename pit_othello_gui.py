import tkinter as tk
from tkinter import messagebox
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import HumanOthelloPlayer

class OthelloGUI:
    def __init__(self, game, player1, player2):
        self.game = game
        self.player1 = player1
        self.player2 = player2
        self.current_player = 1  # Player 1 starts
        self.board = game.getInitBoard()

        # Initialize the GUI
        self.root = tk.Tk()
        self.root.title("Othello Game")
        self.canvas = tk.Canvas(self.root, width=400, height=400, bg="green")
        self.canvas.pack()
        self.draw_board()
        self.canvas.bind("<Button-1>", self.on_click)

        # Start the game loop
        self.play_turn()

    def draw_board(self):
        """Draw the Othello board and pieces."""
        n = self.game.n
        cell_size = 400 // n
        for i in range(n):
            for j in range(n):
                x0, y0 = j * cell_size, i * cell_size
                x1, y1 = x0 + cell_size, y0 + cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black")

                piece = self.board[i][j]
                if piece == 1:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="white")
                elif piece == -1:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="black")

    def on_click(self, event):
        """Handle mouse clicks for human players."""
        if isinstance(self.player1, HumanOthelloPlayer) and self.current_player == 1:
            self.handle_human_move(event, self.player1)
        elif isinstance(self.player2, HumanOthelloPlayer) and self.current_player == -1:
            self.handle_human_move(event, self.player2)

    def handle_human_move(self, event, player):
        """Process a human player's move."""
        n = self.game.n
        cell_size = 400 // n
        x = event.x // cell_size
        y = event.y // cell_size
        action = n * y + x

        valids = self.game.getValidMoves(self.board, self.current_player)
        if valids[action]:
            self.board, self.current_player = self.game.getNextState(self.board, self.current_player, action)
            self.draw_board()
            self.play_turn()
        else:
            messagebox.showerror("Invalid Move", "Please select a valid move.")

    def play_turn(self):
        """Handle the current player's turn."""
        if self.game.getGameEnded(self.board, self.current_player) != 0:
            self.end_game()
            return

        if self.current_player == 1:
            if isinstance(self.player1, HumanOthelloPlayer):
                return  # Wait for human input
            else:
                action = self.player1(self.board)
        else:
            if isinstance(self.player2, HumanOthelloPlayer):
                return  # Wait for human input
            else:
                action = self.player2(self.board)

        self.board, self.current_player = self.game.getNextState(self.board, self.current_player, action)
        self.draw_board()
        self.play_turn()

    def end_game(self):
        """End the game and display the result."""
        result = self.game.getGameEnded(self.board, 1)
        if result == 1:
            messagebox.showinfo("Game Over", "Player 1 (White) wins!")
        elif result == -1:
            messagebox.showinfo("Game Over", "Player 2 (Black) wins!")
        else:
            messagebox.showinfo("Game Over", "It's a draw!")
        self.root.quit()

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()

def main():
    # Initialize the game
    game = OthelloGame(8)

    # Initialize players
    player1 = HumanOthelloPlayer(game).play  # Human player
    player2 = HumanOthelloPlayer(game).play  # Another human player

    # Start the GUI
    gui = OthelloGUI(game, player1, player2)
    gui.run()

if __name__ == "__main__":
    main()