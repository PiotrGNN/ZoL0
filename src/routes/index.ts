import { Application } from 'express';

export function setRoutes(app: Application) {
    const { default: IndexController } = require('../controllers/index');
    const indexController = new IndexController();

    app.get('/', indexController.getIndex.bind(indexController));
}