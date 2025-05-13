import { Request, Response } from 'express';

class IndexController {
    getIndex(req: Request, res: Response) {
        res.send('Welcome to the Express app!');
    }
}

export default IndexController;